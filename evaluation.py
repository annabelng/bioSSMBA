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
from transformers import DistilBertTokenizerFast
import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

def eval_model(cfg: DictConfig):
    #tokenizer = DistilBertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def encode(examples):
         return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

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

    def probability(test_dataset):
        # generates prediction from model
        pred = trainer.predict(test_dataset).predictions
        # softmax each row so each row sums to 1
        prob = softmax(pred, axis = 1)
        # find the mean probability of readmission
        meanprob = np.mean(prob,axis=0)[1]
        # find the max probability of readmission
        maxprob = np.amax(prob,axis=0)[1]

        n = pred.shape[0]

        return meanprob, maxprob, n


    # In[28]:


    def prepare_data(patientID):
        # loading features and labels per patient
        input_dataset = load_dataset('text', data_files={'test': '/h/nng/data/readmit/processPatient/'+patientID})
        label_dataset = load_dataset('text', data_files={'test': '/h/nng/data/readmit/labels/'+patientID})

        # applying encoding function to dataset
        input_dataset = input_dataset.map(encode, batched=True)

        # setting dataset to testing dataset
        test_dataset = Dataset(input_dataset['test'], label_dataset['test'])

        return test_dataset


    # In[31]:


    def readmit_probability(maxprob,meanprob,n):
        # c accounts for patients with many notes
        c=2
        # weight as n/c
        scaling = n/c
        denominator = 1+scaling
        numerator = maxprob + (meanprob * scaling)

        probability = numerator/denominator
        return probability


    # In[7]:


    from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
    model = DistilBertForSequenceClassification.from_pretrained("/h/nng/slurm/2021-01-27/train/checkpoint-12000/")


    # In[32]:


    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    trainer = Trainer(
        model=model                        # the instantiated 🤗 Transformers model to be trained
    )


    # In[ ]:


    # empty list of scalable readmission prediction probabilities
    patient_prob = []

    # generate list of all patients
    patients = os.listdir('/h/nng/data/readmit/processPatient/')

    for i in range(len(patients)):
        # load the patient datset
        test_dataset = prepare_data(patients[i])
        # find the max and mean probability of readmission
        mean, maximum, n = probability(test_dataset)
        readmit = readmit_probability(mean,maximum,n)
        patient_prob.append(readmit)

    print(readmit)
    print(len(patient_prob))



