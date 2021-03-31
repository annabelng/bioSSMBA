#!/usr/bin/env python
# coding: utf-8

# In[1]:


# n number of subsequences of a patient's notes
# c is scaling factor and controls influence of number of subsequences 
# use c= 2
# pmax is max probability of readmission
# pmean is mean probability of readmission
import argparse
from datasets import load_dataset
import os
import numpy as np
import torch
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')


import sklearn.metrics
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, RocCurveDisplay, average_precision_score, precision_recall_curve

def evaluate(real, prob):
    
    # take in probabilities per patient array and threshold
    # turn into list of labels of 0 or 1
    def convert_probability(pred, threshold):
        labels= []
        for val in pred:
            if val > threshold:
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
        pred_prob = readmit_prob
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        roc = roc_auc_score(labels, pred_prob)
        auprc = average_precision_score(labels, pred_prob)
        fpr, tpr, thresholds = roc_curve(labels,pred_prob)
        roc_auc = auc(fpr, tpr)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auroc': roc,
            'auprc' : auprc,
        }

    def compute_tpr(pred_label, real_label, readmit_prob):
        labels = real_label
        preds = pred_label
        predictions = readmit_prob
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        roc = roc_auc_score(labels, predictions)
        fpr, tpr, thresholds = roc_curve(labels,predictions)

        return fpr, tpr

    def compute_prc(pred_label, real_label, readmit_prob):
        labels = real_label
        preds = pred_label
        predictions = readmit_prob
        precision, recall, thresholds = precision_recall_curve(labels, predictions)

        return precision, recall


    def optimizef1 (real_labels, pred_prob):
        test = 0.0
        threshold = 0.0
        optimize = 0.0
        while threshold < 1.0:
            pred_labels = convert_probability(pred_prob, threshold)
            precision, recall, f1, _ = precision_recall_fscore_support(real_labels, pred_labels, average='binary')
            if f1 > test:
                test = f1
                optimize = threshold
            threshold += 0.01
        return optimize

    def load_arrays (label_path, prob_path):
        real_labels = np.load(label_path)
        pred_prob = np.load(prob_path)
        threshold = optimizef1(real_labels, pred_prob)
        pred_labels = convert_probability(pred_prob,threshold)

        return real_labels, pred_prob, pred_labels, threshold 
        
    real_label, pred_prob, pred_label, threshold = load_arrays(real, prob)
    #unique, unique_counts = np.unique(real_label, return_index=False, return_inverse=False, return_counts=True, axis=None)
    #unique2, unique_counts2 = np.unique(pred_label, return_index=False, return_inverse=False, return_counts=True, axis=None)

    print(compute_metrics(pred_label, real_label, pred_prob))
    
    with open('../../data/update/testing/predictions', "w+") as f:
        content = str(pred_prob[:25])
        f.write(content)
    
    with open('../../data/update/testing/real_labels', "w+") as g:
        content = str(real_label[:25])
        g.write(content)
        
    with open('../../data/update/testing/pred_label', "w+") as a:
        content = str(pred_label[:25])
        a.write(content)
    #print(unique_counts)
    #print(unique_counts2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--real', type=str,
                       help='Input path to real labels')
    parser.add_argument('-p', '--prob', type=str,
                       help='Input path to probabilities')
    args = parser.parse_args()

    evaluate(args.real, args.prob)
