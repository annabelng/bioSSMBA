#!/usr/bin/env python
# coding: utf-8

# In[20]:


import csv
import sys
import os
import datetime
import dateutil.parser
#import spacy

# open NOTEEVENTS.csv to extract discharge summaries
d = open("../../mimic/NOTEEVENTS.csv")
reader = csv.reader(d)
header = next(reader)

# open NOTEEVENTS.csv to extract discharge summaries
a = open("../../mimic/ADMISSIONS.csv")
reader2 = csv.reader(a)
header2 = next(reader2)


# In[18]:


elective_list = []
for row in reader2:
    if row[6] == 'ELECTIVE':
        if row[2] in elective_list:
            elective_list.append(row[2])
        else:
            elective_list.append(row[2])


# In[19]:


len(elective_list)


# In[176]:


summaries = {}
for row in reader:
    if row[6] == 'Discharge summary':
        summaries[row[2]] = row[1]


# In[15]:


# function check if note is discharge summary
# outputs patient ID and the text
def process_note_row(row):
    if row[6] == 'Discharge summary':
        feat = []
        feat.append(row[1])
        feat.append(row[2])

        notes = row[-1]
        notes = notes.replace('\n', ' ')

        feat.append(notes)

        return feat


# In[12]:


with open('../../data/update/filtered.txt','r') as f:
    patients = f.readlines()
    filtered = []
    for i in range(len(patients)):
        filtered.append(patients[i].strip('\n'))
        
set_filtered = list(set(filtered))


# In[21]:


for row in reader:
    feat = process_note_row(row)
    if feat:
        ID = feat[0]
        HADM = feat[1]
        if ID not in set_filtered:
            path = '../../data/update/patients/'+ID+'_'+HADM
            with open(path,'w') as g:
                g.write(feat[-1])
        else:
            print (HADM)


# In[ ]:




