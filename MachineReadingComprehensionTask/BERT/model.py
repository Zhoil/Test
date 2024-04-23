from transformers.models.bert.modeling_bert import BertModel
import torch.nn as nn


class BERTQAModel(nn.Module):
    def __init__(self, args):
        super(BERTQAModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.pre_train_model)
        self.classifier = nn.Linear(args.hidden_size, 2)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, start_labels=None,
                end_labels=None):
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids)
        logits = self.classifier(sequence_output['last_hidden_state'])
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        outputs = (start_logits, end_logits,)
        if start_labels is not None and end_labels is not None:
            start_loss = self.loss_fct(start_logits, start_labels)
            end_loss = self.loss_fct(end_logits, end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs
