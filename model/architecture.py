import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial
from model.submodule.bert.bert import MyBertMaskedLM
from utils.pos_embed import get_2d_sincos_pos_embed
# from model.submodule.bert.bert_encoder import BertEncoder
from model.submodule.bert.BertConfig import BertConfig


class MM(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True, mv=False, temp=0.5, c_embed_dim=256):
        super().__init__()

        # --------------------------------------------------------------------------
        # image encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.vision_proj = nn.Linear(embed_dim, c_embed_dim)
        self.text_proj = nn.Linear(embed_dim, c_embed_dim)
        self.itm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, 2)
        )
        # ViT blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # image decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size * 2) ** 2 * in_chans, bias=True)
        # --------------------------------------------------------------------------
        # Bert encoder
        self.bert_encoder, _ = build_text_encoder()
        # self.bert_mlp = nn.Linear(embed_dim, embed_dim, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.fusion_encoder, _ = build_text_encoder()

        self.mv = mv
        if self.mv:
            self.latent_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """

        p = self.patch_embed.patch_size[0] * 2
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0] * 2
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is removed
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_vision_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_vision_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_contrastive_loss(self, latent, caption_ids, labels, attention_mask, token_type_ids, temp=0.5):
        _outputs = self.bert_encoder(None, input_ids=labels, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)  # the feature is generated by fully unmasked
        outputs = _outputs.hidden_states[-1]
        latent = F.normalize(self.vision_proj(latent[:, 0, :]), dim=-1)
        outputs = F.normalize(self.text_proj(outputs[:, 0, :]), dim=-1)
        c_labels = torch.arange(latent.size(0)).type_as(latent).long()
        scores = latent.mm(outputs.t()) / temp
        loss = F.cross_entropy(scores, c_labels) + F.cross_entropy(scores.transpose(0, 1), c_labels)
        return loss, _outputs

    def forward_mlm_loss(self, latent, caption_ids, labels, attention_mask, token_type_ids):
        # latent = self.bert_mlp(latent)
        # # GAP
        # latent = latent[:, 1:, :].mean(dim=1)
        # TODO: maybe try cross attention here
        image_atts=torch.ones(latent.size()[:-1], dtype=torch.long).to(latent.device)
        outputs = self.bert_encoder(latent=None, input_ids=caption_ids, labels=labels, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
        outputs = self.fusion_encoder(latent=None,
                                      inputs_embeds=outputs.hidden_states[-1],
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      labels=labels,
                                      encoder_hidden_states=latent,
                                      encoder_attention_mask=image_atts,  # all ones
                                      return_dict=True)
        # ----------------------------
        # another option, original one
        # outputs = self.bert_encoder(latent=latent, input_ids=caption_ids, labels=labels, attention_mask=attention_mask,
                                    # token_type_ids=token_type_ids)

        # print(len(outputs.hidden_states), outputs.hidden_states[0].shape)
        # print(torch.equal(_.last_hidden_state,outputs.hidden_states[-1]))
        return outputs.loss

    def forward_matching_loss(self, v_embed, outputs_labels, attention_mask, token_type_ids):
        '''
        :param latent:  unmasked vision_embed from vit
        :param outputs_labels:  unmasked text_embed from bert
        :param attention_mask:
        :param token_type_ids:
        :return:
        '''
        bs = v_embed.size(0)
        l_embed = outputs_labels.hidden_states[-1]
        image_atts = torch.ones(v_embed.size()[:-1], dtype=torch.long).to(v_embed.device)
        v_feat = F.normalize(self.vision_proj(v_embed[:, 0, :]), dim=-1)
        l_feat = F.normalize(self.text_proj(l_embed[:, 0, :]), dim=-1)
        output_feat = self.fusion_encoder(latent=None,
                                          inputs_embeds=l_embed,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          encoder_hidden_states=v_embed,
                                          encoder_attention_mask=image_atts,  # all ones
                                          return_dict=True)  # fusion module CA
        with torch.no_grad():
            sim_i2t = v_feat @ l_feat.t() / self.temp
            sim_t2i = l_feat @ v_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(v_embed[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(l_embed[neg_idx])
            text_atts_neg.append(attention_mask[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([l_embed, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, v_embed], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.fusion_encoder(latent=None,
                                         inputs_embeds=text_embeds_all,
                                         attention_mask=text_atts_all,
                                         encoder_hidden_states=image_embeds_all,
                                         encoder_attention_mask=image_atts_all,  # all ones
                                         return_dict=True)

        vl_embeddings = torch.cat([output_feat.hidden_states[-1][:, 0, :], output_neg.hidden_states[-1][:, 0, :]],
                                  dim=0)
        vl_output = self.itm_head(vl_embeddings)  # self.itm_head = nn.Linear(text_width, 2)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(
            v_embed.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_itm

    def forward(self, batch, mask_ratio=0.75):

        # split different views of images
        imgs_1 = batch["image_1"]

        ids, labels, attention_mask, type_ids = batch["ids"], batch["labels"], batch["attention_mask"], batch[
            "type_ids"]

        imgs_1 = imgs_1.cuda()
        ids = ids.cuda()
        labels = labels.cuda()
        attention_mask = attention_mask.cuda()
        type_ids = type_ids.cuda()

        _imgs = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(imgs_1)
        latent, mask, ids_restore = self.forward_vision_encoder(_imgs,
                                                                mask_ratio)  # latent: [N, 50, D], 50=maskratio*196
        latent_unmasked = self.forward_vision_encoder(_imgs, 0.0)[0]  # latent_unmasked: [N, 196, D]
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        v_loss = self.forward_vision_loss(imgs_1, pred, mask)

        if self.mv:
            imgs_2 = batch["image_2"]
            imgs_2 = imgs_2.cuda()
            _imgs_2 = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(imgs_2)
            latent_2, mask_2, ids_restore_2 = self.forward_vision_encoder(_imgs_2, mask_ratio)
            latent_unmasked_2 = self.forward_vision_encoder(_imgs_2, 0.0)[0]

            latent_unmasked = torch.cat((latent_unmasked, latent_unmasked_2), dim=-1)
            latent_unmasked = self.latent_proj(latent_unmasked)

            latent = torch.cat((latent, latent_2), dim=-1)
            latent = self.latent_proj(latent)

            pred_2 = self.forward_decoder(latent_2, ids_restore_2)
            v_loss = 0.5 * v_loss + 0.5 * self.forward_vision_loss(imgs_2, pred_2, mask_2)

        contrastive_loss, outputs_unmasked = self.forward_contrastive_loss(latent_unmasked, ids, labels, attention_mask,
                                                                           type_ids,
                                                                           self.temp)
        mlm_loss = self.forward_mlm_loss(latent, ids, labels, attention_mask, type_ids)
        itm_loss = self.forward_matching_loss(latent_unmasked, outputs_unmasked, attention_mask, type_ids)
        return (v_loss, mlm_loss, itm_loss, contrastive_loss), pred, mask


def build_text_encoder():
    bert_config = BertConfig()
    text_encoder = MyBertMaskedLM(bert_config)
    text_width = text_encoder.config.hidden_size
    return text_encoder, text_width
