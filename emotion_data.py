import pathlib
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer
from sklearn.model_selection import train_test_split

class Emo_Load:
    def __init__(self):
        self.model_name='cl-tohoku/bert-base-japanese-whole-word-masking'
        self.tokenizer=BertJapaneseTokenizer.from_pretrained(self.model_name,is_fast=True)
        self.enc_max_len=30
        self.batch_size=8
        self.input_list=[]
        self.output_list=[]
        
    def emo_data(self):
        p_temp= pathlib.Path('dataset')
        i=0
        sort_temp = sorted(p_temp.glob("*.txt"))
        for p in sort_temp:
          with open(p,'r',encoding='utf-8') as f:
              input_data=f.readlines()
          self.input_list.extend(input_data)
          output_data=[i for x in range(len(input_data))]
          self.output_list.extend(output_data)
          i+=1
        x_train,x_test,t_train,t_test=train_test_split(self.input_list,self.output_list,test_size=0.2, random_state=42, shuffle=True)
        train_data = [(src, tgt) for src, tgt in zip(x_train, t_train)]
        test_data = [(src, tgt) for src, tgt in zip(x_test, t_test)]
        train_ider,test_ider=self.convert_batch_data(train_data,test_data)
        return train_ider,test_ider

    def convert_batch_data(self,train_data, valid_data):
        enc_max_len=self.enc_max_len
        tokenizer=self.tokenizer
        def generate_batch(data):
            batch_src,batch_tgt = [],[]
            for src, tgt in data:
                batch_src.append(src)
                batch_tgt.append(tgt)
            batch_src = tokenizer(batch_src, max_length=enc_max_len, truncation=True, padding="max_length", return_tensors="pt")
            return batch_src,batch_tgt

        train_iter = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=generate_batch)
        valid_iter = DataLoader(valid_data, batch_size=self.batch_size, shuffle=True, collate_fn=generate_batch)

        return train_iter, valid_iter  