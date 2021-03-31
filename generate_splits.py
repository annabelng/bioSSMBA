#!/usr/bin/env python
# coding: utf-8

# In[110]:


import os
import numpy as np
import csv

# open ADMISSIONS.csv to extract discharge summaries
a = open("../../mimic/ADMISSIONS.csv")
reader = csv.reader(a)
header = next(reader)


# In[84]:


with open('../../data/update/splits/filtered_readmit','r') as f:
    lines = f.readlines()
    adm = []
    for i in range(len(lines)):
        adm.append(lines[i].strip('\n'))


# In[111]:


# generate the admit and discharge times for every admission with a discharge summary
# key is patient ID
# value is a list of the admission IDs, admit times, discharge times
admID = {}
for row in reader:
    if row[2] in adm:
        if row[1] in admID:
            admID[row[1]].append(row[2])
        else:
            admID[row[1]] = []
            admID[row[1]].append(row[2])


# In[68]:


# generate list of all patients
admissions = os.listdir('../../data/update/processPatient/')
patientID = []
for i in range(len(admissions)):
    patientID.append(admissions[i].split('_'))

patients = []
for i in range(len(patientID)):
    patients.append(patientID[i][0])

patient_set = set(patients)
patient_list = list(patient_set)


# In[61]:


len(admissions)


# In[69]:


# shuffle patients randomly
shuffle = np.random.permutation(patient_list)


# In[70]:


# roughly a 90-5-5 split
# split into training, validation, and test splits
train = patient_list[:28910]
valid = patient_list[28910:30520]
test = patient_list[30520:]


# In[125]:


admID['6055']


# In[100]:


with open('../../data/update/splits/train_patients','w') as f:
    for line in train:
        f.write(line)
        f.write('\n')
        
with open('../../data/update/splits/valid_patients','w') as f:
    for line in valid:
        f.write(line)
        f.write('\n')
        
with open('../../data/update/splits/test_patients','w') as f:
    for line in test:
        f.write(line)
        f.write('\n')


# In[113]:


with open('../../data/update/splits/train_admissions','w') as f:
    for file in train:
        patient_admissions = admID[file]
        for i in range(len(patient_admissions)):
            path = file + '_' + patient_admissions[i]
            f.write(path)
            f.write('\n')
            
with open('../../data/update/splits/valid_admissions','w') as f:
    for file in valid:
        patient_admissions = admID[file]
        for i in range(len(patient_admissions)):
            path = file + '_' + patient_admissions[i]
            f.write(path)
            f.write('\n')
            
with open('../../data/update/splits/test_admissions','w') as f:
    for file in test:
        patient_admissions = admID[file]
        for i in range(len(patient_admissions)):
            path = file + '_' + patient_admissions[i]
            f.write(path)
            f.write('\n')


# In[114]:


# generate training split for features
with open('../../data/update/splits/train', 'w') as g:
    for file in train:
        patient_admissions = admID[file]
        for i in range(len(patient_admissions)):
            path = file + '_' + patient_admissions[i]
            with open('../../data/update/processPatient/' + path, 'r') as f:
                for line in f:
                    g.write(line)
                g.write('\n')
            print(path)


# In[115]:


# generate test split for features
with open('../../data/update/splits/test', 'w') as g:
    for file in test:
        patient_admissions = admID[file]
        for i in range(len(patient_admissions)):
            path = file + '_' + patient_admissions[i]
            with open('../../data/update/processPatient/' + path, 'r') as f:
                for line in f:
                    g.write(line)
                g.write('\n')
            print(path)


# In[116]:


# generate valid split for features
with open('../../data/update/splits/valid', 'w') as g:
    for file in valid:
        patient_admissions = admID[file]
        for i in range(len(patient_admissions)):
            path = file + '_' + patient_admissions[i]
            with open('../../data/update/processPatient/' + path, 'r') as f:
                for line in f:
                    g.write(line)
                g.write('\n')
            print(path)


# In[117]:


# generate train split for features
with open('../../data/update/splits/train_label','w') as g:
    for file in train:
        patient_admissions = admID[file]
        for i in range(len(patient_admissions)):
            path = file + '_' + patient_admissions[i]
            with open('../../data/update/labels/' + path, 'r') as f:
                for line in f:
                    g.write(line)
                g.write('\n')
            print(path)


# In[118]:


# generate valid split for features
with open('../../data/update/splits/valid_label','w') as g:
    for file in valid:
        patient_admissions = admID[file]
        for i in range(len(patient_admissions)):
            path = file + '_' + patient_admissions[i]
            with open('../../data/update/labels/' + path, 'r') as f:
                for line in f:
                    g.write(line)
                g.write('\n')
            print(path)


# In[119]:


# generate test split for features
with open('../../data/update/splits/test_label','w') as g:
    for file in test:
        patient_admissions = admID[file]
        for i in range(len(patient_admissions)):
            path = file + '_' + patient_admissions[i]
            with open('../../data/update/labels/' + path, 'r') as f:
                for line in f:
                    g.write(line)
                g.write('\n')
            print(path)


# In[120]:


# check line count
print(len(open('../../data/update/splits/train').readlines()))
print(len(open('../../data/update/splits/train_label').readlines()))
print(len(open('../../data/update/splits/test').readlines()))
print(len(open('../../data/update/splits/test_label').readlines()))
print(len(open('../../data/update/splits/valid').readlines()))
print(len(open('../../data/update/splits/valid_label').readlines()))

