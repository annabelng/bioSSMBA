#!/usr/bin/env python
# coding: utf-8

# In[21]:


# import the csv package
import csv

# open ICUSTAYS.csv to see if patient was in neonatal which is abbreviated as NICU
# Used to filter out newborns
b = open("../../mimic/TRANSFERS.csv")
reader2 = csv.reader(b)
header2 = next(reader2)

# open PATIENTS.csv to see if patient died in hospital since it precludes readmission
# filter out those who died
d = open("../../mimic/PATIENTS.csv")
reader3 = csv.reader(d)
header3 = next(reader3)

# filter out newborns
a = open("../../mimic/ADMISSIONS.csv")
reader4 = csv.reader(a)
header4 = next(reader4)


# In[17]:


def filter_baby(row):
    if row[6]=='NEWBORN':
        baby = row[1] 
        return baby
    
def filter_dead(row):
    if row[5]:
        dead = row[1]
        return dead
    
def filter_NICU(row):
    if row[6]=='NICU' or row[7]=='NICU':
        baby = row[1]
        return baby
    elif row[6]=='NWARD' or row[7] =='NWARD':
        baby = row[1]
        return baby


# In[22]:


with open('../../data/update/filtered.txt','w') as f:
    for row in reader2:
        baby = filter_NICU(row)
        if baby:
            f.write(baby+"\n")

    for row in reader4:
        baby = filter_baby(row)
        if baby:
            f.write(baby+"\n")
        dead = filter_dead(row)
        if dead:
            f.write(dead+"\n")


# In[ ]:




