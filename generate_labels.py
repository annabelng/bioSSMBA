#!/usr/bin/env python
# coding: utf-8

# In[58]:


import csv
import sys, os
import datetime
import dateutil.parser

# open NOTEEVENTS.csv to extract discharge summaries
d = open("../../mimic/NOTEEVENTS.csv")
reader = csv.reader(d)
header = next(reader)

# open ADMISSIONS.csv to extract discharge summaries
a = open("../../mimic/ADMISSIONS.csv")
reader2 = csv.reader(a)
header2 = next(reader2)


# In[59]:


summaries = {}
for row in reader:
    if row[6] == 'Discharge summary':
        summaries[row[2]] = row[1]


# In[56]:


readmit = {}
for row in reader2:
    if row[2] in readmit:
        readmit[row[2]].append(row[1])
    else:
        readmit[row[2]] = []
        readmit[row[2]].append(row[1])


# In[7]:


with open('../../data/update/filtered.txt','r') as f:
    patients = f.readlines()
    filtered = []
    for i in range(len(patients)):
        filtered.append(patients[i].strip('\n'))

set_filtered = set(filtered)


# In[39]:


# filtering out the discharge summaries of dead people and babies
filtered_summaries = {}
for key in summaries:
    if summaries[key] not in set_filtered:
        filtered_summaries[key] = summaries[key]

# find how many admissions do not have a discharge summary
missing = {}
existing_summaries = {}
for key in readmit:
    if key not in filtered_summaries:
        missing[key] = readmit[key][0]
    else:
        existing_summaries[key] = filtered_summaries[key]
        
# of the missing discharge summaries, see which match with a filtered patient
filtered_missing = {}
for key in missing:
    if missing[key] not in set_filtered:
        filtered_missing[key] = missing[key]

# check how many existing admissions there are after filtering
filtered_readmit = {}
for key in readmit:
    if readmit[key][0] not in set_filtered:
        if key not in filtered_missing:
            filtered_readmit[key] = readmit[key]


# In[19]:


with open('../../data/update/splits/filtered_readmit','w') as f:
    for key in filtered_readmit:
        f.write(key)
        f.write('\n')


# In[43]:


# generate the admit and discharge times for every admission with a discharge summary
# key is patient ID
# value is a list of the admission IDs, admit times, discharge times
times = {}
for row in reader2:
    if row[2] in filtered_readmit:
        if row[1] in times:
            times[row[1]].append(row[2])
            times[row[1]].append(row[3])
            times[row[1]].append(row[4])
        else:
            times[row[1]] = []
            times[row[1]].append(row[2])
            times[row[1]].append(row[3])
            times[row[1]].append(row[4])


# In[44]:


len(times)


# In[57]:


len(readmit)


# In[53]:


len(readmit)


# In[54]:


len(readmit2)


# In[26]:


len(times2)


# In[101]:


len(filtered_readmit)


# In[102]:


len(existing_summaries)


# In[42]:


def check_readmit(times):
    readmit_label = []
    if len(times)>2:
        i=2
        while i < len(times)-1:
            discharge = dateutil.parser.parse(times[i])
            readmit = dateutil.parser.parse(times[(i+2)])
            readmit_cutoff = discharge + datetime.timedelta(days=30)
            if readmit < readmit_cutoff:
                print('30 day readmit')
                readmit_label.append([1,times[i-2]])
            else:
                print('over 30 days')
                readmit_label.append([0,times[i-2]])
            i += 3
        readmit_label.append([0,times[i-2]])
    else:
        print ('no readmit')
        readmit_label.append([0,times[0]])
    
    return readmit_label


# In[150]:


def write_label(readmit_label,patient_ID):
    for i in range(len(readmit_label)):
        adm_ID = readmit_label[i][1]
        path = patient_ID + '_' + adm_ID
        label = readmit_label[i][0]
        with open('../../data/update/processPatient/'+ path,'r') as f:
            with open('../../data/update/labels/'+ path,'w') as g:
                lines = len(f.readlines())
                for i in range(lines):
                    if i == (lines-1):
                        if label == 1:
                            g.write('1')
                        else:
                            g.write('0')
                    else:
                        if label == 1:
                            g.write('1\n')
                        else:
                            g.write('0\n')


# In[140]:


admissions = os.listdir('../../data/update/processPatient/')
patientID = []
for i in range(len(admissions)):
    patientID.append(admissions[i].split('_'))

patients = []
for i in range(len(patientID)):
    patients.append(patientID[i][0])

patient_set = set(patients)
patient_list = list(patient_set)


# In[142]:


len(patient_list)


# In[151]:


for i in range(len(patient_list)):
    test_patient = patient_list[i]
    test_readmit = check_readmit(times[test_patient])
    write_label(test_readmit, test_patient)


# In[14]:


readmit_count = 0
for i in patientList:
    if labels[i]>1:
        readmit_count += 1
        


# In[16]:


readmit_count


# In[17]:


len(patientList)


# In[24]:


readmit['7101']


# In[145]:


check_date(times['7101'])


# In[45]:


test_patient = patientID[20][0]
test_readmit = check_readmit(readmit[test_patient])
write_label(test_readmit, test_patient)


# In[54]:


check_readmit(readmit['23657'])


# In[159]:


times['36']


# In[72]:


len(readmit)


# In[152]:


label_path = os.listdir('../../data/update/labels/')
positive = 0
negative = 0
for i in range(len(label_path)):
    with open('../../data/update/labels/'+label_path[i],'r') as f:
        test = f.readline()
        if test == '1' or test == '1\n':
            positive += 1
        else:
            negative += 1


# In[153]:


print(positive)
print(negative)


# In[154]:


len(filtered_missing)


# In[ ]:




