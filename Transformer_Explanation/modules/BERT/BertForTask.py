from transformers import BertPreTrainedModel
from transformers.utils import logging
from Transformer_Explanation.modules.layers_ours import *
from Transformer_Explanation.modules.BERT.BERT import BertModel
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
from typing import List, Any
import torch
from transformers.modeling_outputs import SequenceClassifierOutput, QuestionAnsweringModelOutput


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def relprop(self, cam=None, **kwargs):
        cam = self.classifier.relprop(cam, **kwargs)
        cam = self.dropout.relprop(cam, **kwargs)
        cam = self.bert.relprop(cam, **kwargs)
        # print("conservation: ", cam.sum())
        return cam


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions = None,
            end_positions = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return SequenceClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def relprop(self, cam=None, alpha=1, index=None):
        kwargs = {'alpha': alpha}
        cam = self.qa_outputs.relprop(cam, **kwargs)
        if index is not None:
            cam = cam[:, index, :].reshape(1, -1)
        cam = self.bert.relprop(cam, **kwargs)
        return cam




if __name__ == '__main__':
    from transformers import BertTokenizer
    import os

    class Config:
        def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, num_labels,
                     hidden_dropout_prob):
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.num_labels = num_labels
            self.hidden_dropout_prob = hidden_dropout_prob


    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    x = tokenizer.encode_plus("In this movie the acting is great. The movie is perfect! [sep]",
                         add_special_tokens=True,
                         max_length=512,
                         return_token_type_ids=False,
                         return_attention_mask=True,
                         pad_to_max_length=True,
                         return_tensors='pt',
                         truncation=True)

    print(x['input_ids'])

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model_save_file = os.path.join('./Transformer_Explanation/output_bert/movies/classifier/', 'classifier.pt')
    model.load_state_dict(torch.load(model_save_file))

    # x = torch.randint(100, (2, 20))
    # x = torch.tensor([[101, 2054, 2003, 1996, 15792, 1997, 2023, 3319, 1029, 102,
    #                    101, 4079, 102, 101, 6732, 102, 101, 2643, 102, 101,
    #                    2038, 102, 101, 1037, 102, 101, 2933, 102, 101, 2005,
    #                    102, 101, 2032, 102, 101, 1010, 102, 101, 1037, 102,
    #                    101, 3800, 102, 101, 2005, 102, 101, 2010, 102, 101,
    #                    2166, 102, 101, 1010, 102, 101, 1998, 102, 101, 2010,
    #                    102, 101, 4650, 102, 101, 1010, 102, 101, 2002, 102,
    #                    101, 2074, 102, 101, 2515, 102, 101, 1050, 102, 101,
    #                    1005, 102, 101, 1056, 102, 101, 2113, 102, 101, 2054,
    #                    102, 101, 1012, 102]])
    # x.requires_grad_()

    model.eval()

    y = model(x['input_ids'], x['attention_mask'])
    print(y)

    cam, _ = model.relprop()

    #print(cam.shape)

    cam = cam.sum(-1)
    #print(cam)
