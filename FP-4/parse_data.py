#!/usr/bin/env python
# coding: utf-8

# # parse_data.ipynb
# 
# This notebook parses the data files used for the FP-2 assignment. 
# 
# <br>
# <br>
# 
# First let's read the attached data file:

# In[1]:


import pandas as pd

df0 = pd.read_csv('lifestylestudents.csv')
df0.head()
df0.describe(include='all')


# <br>
# <br>
# 
# The dependent and independent variables variables (DVs and IVs) that we are interested in are:
# 
# **DVs**:
# - grades
# 
# **IVs**:
# - daily study hours
# - daily extracurricular hours
# - daily hours of sleep
# - daily social hours
# - daily hours of physical activity
# 
# <br>
# <br>
# 
# Let's extract the relevant columns:

# In[2]:


df = df0[  ['Grades', 'Stress_Level', 'Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day']  ]
df.head()
df.describe(include='all')


# <br>
# <br>
# 
# Next let's use the `rename` function to give the columns simpler variable names:

# In[3]:


df = df.rename( columns={'Grades':'grades', 'Stress_Level':'stress', 'Study_Hours_Per_Day':'studyhours', 'Extracurricular_Hours_Per_Day':'echours', 'Sleep_Hours_Per_Day':'sleephours', 'Social_Hours_Per_Day':'socialhours', 'Physical_Activity_Hours_Per_Day':'activityhours'} )
df['stress'] = df['stress'].map({'Low': 1, 'Moderate': 2, 'High': 3})
df.describe(include='all')

import pandas as pd

df = pd.read_csv("lifestylestudents.csv") 

def parse_data():
    return df

    
df = df.rename(columns={
    'Study_Hours_Per_Day': 'studyhours',
    'Extracurricular_Hours_Per_Day': 'echours',
    'Sleep_Hours_Per_Day': 'sleephours',
    'Social_Hours_Per_Day': 'socialhours',
    'Physical_Activity_Hours_Per_Day': 'activityhours',   # <-- FIXED
    'Grades': 'grades',
    'Stress_Level': 'stress'
})

