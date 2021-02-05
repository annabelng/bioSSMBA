#!/usr/bin/env python
# coding: utf-8

# In[56]:

import argparse
from datasets import load_dataset
import os
import numpy as np
import torch
from scipy.special import softmax


# In[57]:

def predict(args):
    from transformers import DistilBertTokenizerFast
    #tokenizer = DistilBertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


    # In[62]:


    from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
    model = DistilBertForSequenceClassification.from_pretrained(args.model)


    # In[58]:


    def encode(examples):
         return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)


    # In[86]:


    # training model on tokenized and split data
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, inputs):
            self.inputs = inputs

        def __getitem__(self, idx):
            item = {key: torch.tensor(val) for key, val in self.inputs[idx].items() if key != 'text'}
            return item

        def __len__(self):
            return len(self.inputs)


    # In[87]:


    def prepare_data(filePath):
        # loading features and labels per patient
        input_dataset = load_dataset('text', data_files={'test': filePath})

        # applying encoding function to dataset
        input_dataset = input_dataset.map(encode, batched=True)

        # initiating dataset object
        test_dataset = Dataset(input_dataset['test'])

        return test_dataset


    # In[63]:


    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    trainer = Trainer(
        model=model                        # the instantiated 🤗 Transformers model to be trained
    )


    # In[85]:


    # load the patient datset with file path
    test_dataset = prepare_data(args.input)

    # generate probability of readmission
    pred = trainer.predict(test_dataset).predictions

    # softmax each row so each row sums to 1
    prob = softmax(pred, axis = 1)

    # write the probabilities out to text file
    with open(args.output,'w') as f:
        for row in prob:
            if args.soft:
                for column in row:
                    f.write(str(column))
                    f.write(" ")
                f.write('\n')
            else:
                pred = np.argmax(row)
                f.write(str(pred) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input file')
    parser.add_argument('-m', '--model', type=str, help='model file')
    parser.add_argument('-o', '--output', type=str, help='output file')
    parser.add_argument('-s', '--soft', action='store_true')
    args = parser.parse_args()
    predict(args)


