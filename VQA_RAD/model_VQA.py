import numpy as np
import torch
import torchvision
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from model.archi import MM
from model.submodule.bert.BertConfig import BertConfig
from model.submodule.bert.bert import BertLMHeadModel


class MyVQAModel(MM):
    def __init__(self):
        super(MyVQAModel, self).__init__()
        self.bert_decoder = BertLMHeadModel(config=BertConfig(is_decoder=True,add_cross_attention=True))
        self.initialize_weights()
    def forward(self, images, input_ids, attention_mask, answer, answer_attention, train:bool=True):
        """
        :param images:
        :param input_ids: question ids
        :param attention_mask: question attention mask
        :param answer: answer ids
        :param answer_attention: answer attention mask
        :param train: bool
        :return: loss/ topk_ids, topk_probs depends on train
        """
        image=torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(images)
        image_embeds, _ = self.forward_vision_encoder(image,0.0)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        if train:
            answer_targets = answer.masked_fill(answer == 0, -100)
            question_output = self.bert_encoder(input_ids,
                                                attention_mask=attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            answer_output = self.bert_decoder(answer,
                                              attention_mask=answer_attention,
                                              encoder_hidden_states=question_output.last_hidden_state,
                                              encoder_attention_mask=attention_mask,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )
            loss = answer_output.loss
            loss = loss.sum() / images.size(0)   # images.size(0) = batch_size
            return loss

        # test
        else:
            question_output = self.bert_encoder(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            topk_ids, topk_probs = self.rank_answer(question_output.hidden_states[-1], attention_mask,
                                                    answer, answer_attention)

            return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k=64):
        def tile(x, dim, n_tile):
            init_dim = x.size(dim)
            repeat_idx = [1] * x.dim()
            repeat_idx[dim] = n_tile
            x = x.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
            return torch.index_select(x, dim, order_index.to(x.device))

        num_ques = question_states.size(0)  # num_question = batch_size_test
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.bert_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)

        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))

        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == 0, -100)
        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.bert_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)
        topk_probs = F.softmax(log_probs_sum, dim=-1)

        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)
        return topk_ids, topk_probs

