from transformers.models.bert.modeling_bert import BertModel
import torch.nn as nn


class BERTClassificationModel(nn.Module):
    def __init__(self, args):
        super(BERTClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.pre_train_model)
        self.classifier = nn.Linear(args.hidden_size, args.label_number)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, labels=None):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     position_ids=position_ids)['pooler_output']
        logits = self.classifier(pooled_output)
        score = nn.functional.softmax(logits, dim=-1)
        outputs = (score,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs
        return outputs
