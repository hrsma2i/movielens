#!/usr/bin/env python


# coding: utf-8

# In[ ]:

from recommendation import *


# In[ ]:

model = SVD_SGD()
model.load_params('results/test/parameters.pkl')


# In[ ]:

# check top-k rated items
user = 0
k = 10
recommend_topk(R=model.R, u=user, k=k)


# In[ ]:

# recommend top-k unrated items
mask_unrated = (model.R[user]==0)
recommend_topk(R=model.get_R_p(), u=user, k=k, mask=mask_unrated)


# In[ ]:



