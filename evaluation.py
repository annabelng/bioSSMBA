#!/usr/bin/env python
# coding: utf-8

# In[2]:


# n number of subsequences of a patient's notes
# c is scaling factor and controls influence of number of subsequences
# use c= 2
# pmax is max probability of readmission
# pmean is mean probability of readmission
from datasets import load_dataset
import os
import numpy as np
import torch
from scipy.special import softmax
import json
from omegaconf import DictConfig
import hydra
from hydra import slurm_utils
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments


# In[1]:
@hydra.main(config_path='/h/nng/conf/biossmba', config_name='config')
def eval_model(cfg: DictConfig):

    tokenizer = DistilBertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    #tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


    # In[3]:


    def encode(examples):
         return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)


    # In[4]:


    if cfg.extra == 'base':
        model = DistilBertForSequenceClassification.from_pretrained("/checkpoint/nng/keep/train_lr2e-05/checkpoint-6000")
    elif cfg.noise == 0.1 and cfg.label == 'soft':
        model = DistilBertForSequenceClassification.from_pretrained("/h/nng/slurm/2021-03-04/train_ssmba_0.1_soft/2077547/checkpoint-31000")
    elif cfg.noise == 0.1 and cfg.label == 'pres':
        model = DistilBertForSequenceClassification.from_pretrained("/checkpoint/nng/keep/train_ssmba_0.1_pres/checkpoint-31000")
    elif cfg.noise == 0.2 and cfg.label == 'soft':
        model = DistilBertForSequenceClassification.from_pretrained("/h/nng/slurm/2021-03-04/train_ssmba_0.2_soft/2077549/checkpoint-31000")
    elif cfg.noise == 0.2 and cfg.label == 'pres':
        model = DistilBertForSequenceClassification.from_pretrained("/checkpoint/nng/keep/train_ssmba_0.2_pres/checkpoint-31000")
    elif cfg.noise == 0.3 and cfg.label == 'soft':
        model = DistilBertForSequenceClassification.from_pretrained("/h/nng/slurm/2021-03-04/train_ssmba_0.3_soft/2077551/checkpoint-31000")
    elif cfg.noise == 0.3 and cfg.label == 'pres':
        model = DistilBertForSequenceClassification.from_pretrained("/checkpoint/nng/keep/train_ssmba_0.3_pres/checkpoint-31000")
    elif cfg.noise == 0.4 and cfg.label == 'soft':
        model = DistilBertForSequenceClassification.from_pretrained("/h/nng/slurm/2021-03-04/train_ssmba_0.4_soft/2077553/checkpoint-31000")
    elif cfg.noise == 0.4 and cfg.label == 'pres':
        model = DistilBertForSequenceClassification.from_pretrained("/checkpoint/nng/keep/train_ssmba_0.4_pres/checkpoint-31000")


    # In[5]:
    model.eval()
    model.cuda()

    # training model on tokenized and split data
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val) for key, val in self.inputs[idx].items() if key != 'text'}
            item['labels'] = torch.tensor(int(self.labels[idx]['text']))
            return item

        def __len__(self):
            return len(self.labels)


    # In[6]:


    def probability(test_dataset):
        # generates prediction from model
        train_pred = trainer.predict(test_dataset)
        pred = train_pred.predictions

        # softmax each row so each row sums to 1
        prob = softmax(pred, axis = 1)[:,1]
        prob_list = prob.tolist()

        # find the mean probability of readmission
        meanprob = np.mean(prob,axis=0)[1]

        # find the max probability of readmission
        maxprob = np.amax(prob,axis=0)[1]

        # number of subsequences
        n = pred.shape[0]
        
        # putting mean, max, and n into list
        mean_max_n = []
        mean_max_n.extend((meanprob, maxprob, n))

        # return mean, max, shape
        return prob_list, mean_max_n


    # In[7]:


    def prepare_data(patientID):
        # loading features and labels per patient
        input_dataset = load_dataset('text', data_files={'test': '/h/nng/data/readmit/processPatient/'+patientID})
        label_dataset = load_dataset('text', data_files={'test': '/h/nng/data/readmit/labels/'+patientID})

        # applying encoding function to dataset
        input_dataset = input_dataset.map(encode, batched=True)

        # setting dataset to testing dataset
        test_dataset = Dataset(input_dataset['test'], label_dataset['test'])

        return test_dataset


    # In[8]:


    # calculating readmit probability on per patient basis
    def readmit_probability(maxprob,meanprob,n):
        # c accounts for patients with many notes
        c=2
        # weight as n/c
        scaling = n/c
        denominator = 1+scaling
        numerator = maxprob + (meanprob * scaling)

        probability = numerator/denominator
        return probability


    # In[9]:


    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    # generating numpy array of all the real labels
    def patient_labels(patients):
        labels = []
        for i in range(len(patients)):
            # taking label per patient
            with open('/h/nng/data/readmit/labels/'+ patients[i], 'r') as f:
                text = f.readline().strip()
                if text == '1':
                    labels.append(1)
                elif text == '0':
                    labels.append(0)

        label_array = np.asarray(labels)

        return label_array

    # take in probabilities per patient array and threshold
    # turn into list of labels of 0 or 1
    def convert_probability(pred, threshold):
        labels= []
        for val in pred:
            if val>threshold:
                labels.append(1)
            else:
                labels.append(0)

        labels_array = np.asarray(labels)
        return labels_array

    # computing accuracy, f1, precision, recall, auroc
    # parameters are the arrays of predicted labels, real labels, and predicted probabilities
    def compute_metrics(pred_label, real_label, readmit_prob):
        labels = real_label
        preds = pred_label
        predictions = readmit_prob
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        roc = roc_auc_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auroc': roc,
        }
    
    # calculating readmit probability on per patient basis
    # trying different c values
    '''def optimize_c(raw_prob):
        for i in raw_prob:
            
        # c accounts for patients with many notes
        c=2
        # weight as n/c
        scaling = n/c
        denominator = 1+scaling
        numerator = maxprob + (meanprob * scaling)

        probability = numerator/denominator
        return probability'''


    # In[10]:


    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
    )


    # In[11]:


    with open('/h/nng/data/readmit/mimic/orig/valid_admissions','r') as f:
        lines = f.read().splitlines()
        set_valid = set(lines)
    valid_list = list(set_valid)

    with open('/h/nng/data/readmit/mimic/orig/test_admissions','r') as f:
        lines = f.read().splitlines()
        set_test = set(lines)
    test_list = list(set_test)


    # In[22]:


    # takes in list of patients from either valid split or test split
    # lists are valid_list or test_list
    def evaluate(split):
        # empty list of scalable readmission prediction probabilities
        patient_prob = []
        patient_mean_max = []

        # load valid list for testing
        for i in range(len(split)):
            # load the patient datset
            test_dataset = prepare_data(split[i])

            # find the max and mean probability of readmission
            raw_prob, mean_max_n = probability(test_dataset)

            # calculate readmission probability per patient
            #readmit = readmit_probability(mean,maximum,n)

            # add probabilities into list of all patient probabilities
            patient_prob.append(raw_prob)
            patient_mean_max.append(mean_max_n)
            #print(i)

        return patient_prob, patient_mean_max


    # In[20]:


    # generating patient probability from model
    # pass in either valid_list or test_list
    patient_prob, patient_mean_max = evaluate(test_list)
    mean_max_vals = np.asarray(patient_mean_max)

    # generating actual labels of patients for valid list
    # pass in either valid_list or test_list
    real_labels = patient_labels(test_list)

    # turn predicted probability list into 1d numpy array
    #pred_prob = np.asarray(patient_prob)

    # generate label array from probability list and threshold
    # if probability over a certain threshold, generate a readmit label of 1
    # otherwise, readmit = 0
    #pred_labels = convert_probability(pred_prob,0.5)

    #print(real_labels)
    #print(pred_prob)
    #print(pred_labels)


    j_dir = slurm_utils.get_j_dir(cfg)
    o_dir = os.path.join(j_dir, os.environ['SLURM_JOB_ID'])

    with open(os.path.join(o_dir, 'pred_prob.json'), 'w') as outfile:
        json.dump(patient_prob, outfile)

    with open(os.path.join(o_dir, 'mean_max_n.npz'), 'wb') as f:
        np.save(f, mean_max_vals)
        
    #with open(os.path.join(o_dir, 'pred_labels.npz'), 'wb') as f:
    #    np.save(f, pred_labels)
    with open(os.path.join(o_dir, 'real_labels.npz'), 'wb') as f:
        np.save(f, real_labels)

    # computing the metrics
    #print(compute_metrics(pred_labels, real_labels,pred_prob))

if __name__ == "__main__":
    eval_model()
