#!/usr/bin/env python
# coding: utf-8

# In[1]:


import climate_job_maintenance as cjm

import os 
import numpy as np
'''os.system("jupyter nbconvert --to script 'climate_train.ipynb'")
os.system("jupyter nbconvert --to script 'climate_data.ipynb'")
os.system("jupyter nbconvert --to script 'climate_models.ipynb'")
os.system("jupyter nbconvert --to script 'climate_job_maintenance.ipynb'")'''

import climate_train as ct
import climate_data
import climate_models
import climate_job_maintenance as cjm




# In[4]:


args=ct.options(string_input="-b 3 --depth 1".split())

args.model_id=3
args.model_bank_id="G"

C,names=climate_models.golden_model_bank(args,only_description=True,verbose=True)


# In[7]:


offset=0
x=[[0],[0,1],[1],np.arange(6).tolist(),np.arange(4).tolist()]
J=cjm.jobnums(C=C,x=x,offset=offset)


# In[8]:


print(str(J).replace(' ','')),len(J)


# In[ ]:


cjm.configure_models(J)

