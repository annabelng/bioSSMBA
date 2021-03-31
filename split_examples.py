#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import os,glob
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


# In[2]:


def tokenize(patientID):
    with open('../../data/update/patients/'+patientID,'r') as f:
        with open('../../data/update/processPatient/'+patientID,'w') as g:
            test = ''
            newLength = 0
            i = 220           
            for line in f:
                words = line.split(" ")
                chunk = " ".join(words[0:i])
                g.write(chunk+" ")
                test += chunk + " "
                while i < len(words):
                    if newLength >= 475:
                        g.write('\n'+ words[i] +" ")
                        test = words[i] + " "
                        token = tokenizer.encode(test)
                        newLength = len(token)
                        i += 1
                    else:
                        chunk = " ".join(words[i:(i+5)])
                        g.write(chunk+" ")
                        test += chunk + " "
                        token = tokenizer.encode(test)
                        newLength = len(token)
                        i += 5
                else:
                    continue
                break


# In[4]:


# open patients text file
patientList = os.listdir('../../data/update/patients/')
for i in range(len(patientList)):
    tokenize(patientList[i])
    print(patientList[i])


# In[4]:


patientList = os.listdir('../../data/update/patients/')


# In[5]:


patientList[0]


# In[ ]:




