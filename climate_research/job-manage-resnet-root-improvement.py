#!/usr/bin/env python
# coding: utf-8

# In[1]:


import climate_job_maintenance as cjm

import os 
import numpy as np
'''os.system("jupyter nbconvert --to script 'climate_train.ipynb'")
'''
os.system("jupyter nbconvert --to script 'climate_models.ipynb'")
os.system("jupyter nbconvert --to script 'climate_data.ipynb'")
#os.system("jupyter nbconvert --to script 'climate_job_maintenance.ipynb'")

import climate_train as ct
import climate_data
import climate_models
import climate_job_maintenance as cjm


# In[2]:


args=ct.options(string_input="-b 3 --depth 0".split())

args.model_id=9000
args.model_bank_id="G"

C,names=climate_models.golden_model_bank(args,only_description=True,verbose=True)


# In[3]:


offset=9000
#testing global or 4reg training
x=[[0,1],[0,1],[0],[0],[1],[0,1,2]]
J=cjm.jobnums(C=C,x=x,offset=offset)
#testing res
x=[[1],[1],[0,1],[1],[1],[0,1,2]]
J+=cjm.jobnums(C=C,x=x,offset=offset)
#


# In[4]:


print(str(J).replace(' ','')),len(J)


# In[5]:


cjm.configure_models(J)


# In[17]:


x=[[0,1],[0,1],[0],[0],np.arange(1).tolist(),[0,1,2]]
cjm.report_progress(C,x,offset=offset)


# In[8]:


x=[[1],[1],[0,1],[1],np.arange(1).tolist(),[0,1,2]]
cjm.report_progress(C,x,offset=offset)


# In[ ]:


'''cjm.report_progress(C,x,offset=offset)'''


