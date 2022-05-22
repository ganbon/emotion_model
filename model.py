import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertForSequenceClassification,BertJapaneseTokenizer


class Bertclass_model(nn.Module):
    def __init__(self):
        super(Bertclass_model,self).__init__()
        self.model_name='cl-tohoku/bert-base-japanese-whole-word-masking'
        self.tokenizer=BertJapaneseTokenizer.from_pretrained(self.model_name)
        self.model=BertForSequenceClassification.from_pretrained(self.model_name,num_labels=5)
        
    def forward(self,input_ids,attention_mask=None,decoder_input_ids=None,decoder_attention_mask=None,labels=None):
        output=self.model(input_ids,labels=labels)
        return output
        