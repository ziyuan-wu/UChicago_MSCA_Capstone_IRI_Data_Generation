#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('store_drug_combined_total.csv')


# In[4]:


from sdv.tabular import CTGAN

import timeit

start = timeit.default_timer()


model = CTGAN()
model.fit(df)


stop = timeit.default_timer()

print('Time: ', stop - start)  


# In[ ]:


new_data = model.sample(num_rows=10000)


# In[ ]:


model.save('simple_model.pkl')


# In[ ]:





# In[ ]:




