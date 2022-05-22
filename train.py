import torch
from tqdm import tqdm
from model import Bertclass_model
from emotion_data import Emo_Load
import copy
from pathlib import Path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Bertclass_model()
epoch=10
best_loss = float('Inf')
pad=model.tokenizer.pad_token_id
emo=Emo_Load()
train,test=emo.emo_data()
optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001)
for e in range(epoch):
    model.train()
    train_losses=0
    pbar=tqdm(train)
    for data,target in pbar:
        optimizer.zero_grad()
        labels=torch.tensor(target).unsqueeze(0)
        output=model(data["input_ids"].to(device),
                          attention_mask=data["attention_mask"].to(device),
                          labels=labels.to(device))
        loss=output.loss
        loss.backward()
        optimizer.step()
        train_losses += loss
        pbar.set_postfix(loss=train_losses/epoch)
        loss_train=train_losses /len(data)
    model.eval()
    test_losses = 0
    with torch.no_grad():
        for data, target in test:
            labels = torch.tensor(target)
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                labels=labels.to(device)
            )
            loss,logits=output[:2]
            test_loss = loss
            test_losses += loss
    if best_loss > test_loss:
        best_loss = test_loss
        best_model = copy.deepcopy(model)
        counter = 1
        
model_dir_path = Path('model_bert')
if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)

model.tokenizer.save_pretrained(model_dir_path)
best_model.model.save_pretrained(model_dir_path)   
