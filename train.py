#!/usr/bin/env python
# coding: utf-8

# In[26]:


from pathlib import Path

def read_data_split(file):
    texts = []
    with open(file,'r') as f:
        for line in f:
            texts.append(line.strip())
    return texts

def read_label_split(file):
    labels = []
    with open(file,'r') as f:
        for line in f:
            if line == '1\n':
                labels.append(1)
            else:
                labels.append(0)
    return labels

train_texts = read_data_split('../data/splits/train2')
train_labels = read_label_split('../data/splits/train_label2')
val_texts = read_data_split('../data/splits/valid2')
val_labels = read_label_split('../data/splits/valid_label2')
test_texts = read_data_split('../data/splits/test2')
test_labels = read_label_split('../data/splits/test_label2')


# In[27]:


from transformers import DistilBertTokenizerFast
#tokenizer = DistilBertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# In[28]:


train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


# In[29]:


len(test_encodings['input_ids'])


# In[30]:


# training model on tokenized and split data
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)
test_dataset = Dataset(test_encodings, test_labels)


# In[ ]:


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='../data/results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()




