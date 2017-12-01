#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display

from recommendation import *


# In[ ]:

model = SVD_SGD()
model.load_params('results/sgd/parameters.pkl')


# In[ ]:

# check top-k rated items
user = 0
k = 10
recommend_topk(R=model.R, u=user, k=k)


# In[ ]:

# recommend top-k unrated items
mask_unrated = (model.R[user]==0)
rec = recommend_topk(R=model.get_R_p(), u=user, k=k, mask=mask_unrated)
display(rec)

log_json = 'results/sgd/log'
df_log = pd.read_json(log_json)
for name, col in df_log.iteritems():
    plt.plot(col, label=name)
plt.legend()


# In[ ]:

# recommend top-k unrated items
model = SVD_SGD(biased=True)
model_name = 'sgd_b'

model.load_params('results/{}/parameters.pkl'.format(model_name))
mask_unrated = (model.R[user]==0)
rec = recommend_topk(R=model.get_R_p(), u=user, k=k, mask=mask_unrated)
display(rec)

log_json = 'results/{}/log'.format(model_name)
df_log = pd.read_json(log_json)
for name, col in df_log.iteritems():
    plt.plot(col, label=name)
plt.legend()


# In[ ]:

# recommend top-k unrated items
model = SVD_ALS()
model_name = 'als'

model.load_params('results/{}/parameters.pkl'.format(model_name))
mask_unrated = (model.R[user]==0)
rec = recommend_topk(R=model.get_R_p(), u=user, k=k, mask=mask_unrated)
display(rec)

log_json = 'results/{}/log'.format(model_name)
df_log = pd.read_json(log_json)
for name, col in df_log.iteritems():
    plt.plot(col, label=name)
plt.legend()


# In[ ]:

# recommend top-k unrated items
model = SVD_ALS(biased=True)
model_name = 'als_b'

model.load_params('results/{}/parameters.pkl'.format(model_name))
mask_unrated = (model.R[user]==0)
rec = recommend_topk(R=model.get_R_p(), u=user, k=k, mask=mask_unrated)
display(rec)

log_json = 'results/{}/log'.format(model_name)
df_log = pd.read_json(log_json)
for name, col in df_log.iteritems():
    plt.plot(col, label=name)
plt.legend()


# In[ ]:



