import torch.nn as nn

from .BertConfig import BertConfig
from .bert import MyBertMaskedLM


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()

        self.model = MyBertMaskedLM(BertConfig())

    def forward(self, latent, ids, labels, attn_mask, token_type,
                encoder_hidden_states=None, encoder_attention_mask=None):
        # option 1: use the latent addition way to fusion
        outputs = self.model(latent=latent, input_ids=ids, attention_mask=attn_mask,
                             token_type_ids=token_type, labels=labels,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)

        # option 2: use the cross-attention way to fusion
        # TODO: in this way, need to assign the encoder_hidden_states and encoder_attention_mask in the forward function.
        return outputs
