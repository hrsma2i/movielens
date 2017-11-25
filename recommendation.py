
# coding: utf-8

# In[ ]:


from copy import deepcopy
import os
import codecs
from itertools import product

from IPython.display import display
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:


def rms(arr):
    return np.sqrt(np.mean((arr**2).ravel()))


# In[ ]:


def SVD(R, n_epochs=20, K=100, lr=0.05, reg=0.02):
    # U: the number of users
    # I: the number of items
    U, I = R.shape
    # P: user factor matrix
    P = np.random.rand(K, U)
    # Q: item factor matrix
    Q = np.random.rand(K, I)
    
    # only use rated elements of R to compute loss
    mask = np.ones(R.shape)
    mask[R == 0] = 0
    
    # learning
    for epoch in trange(n_epochs, desc='epoch'):
        pbar = tqdm(product(range(U), range(I)),
                 total=U*I, desc='(u, i)')
        for u, i in pbar:
            # only update rated elements
            Rui = R[u,i]
            if Rui > 0:
                # compute error
                eui = Rui - P[:,u].T.dot(Q[:,i])
                for k in range(K):
                    Pku = P[k,u]
                    Qki = Q[k,i]
                    P[k,u] += lr * (2 * eui * Qki - reg * Pku)
                    Q[k,i] += lr * (2 * eui * Pku - reg * Qki)
                    
        # compute & print loss
        E = R - (P.T.dot(Q))*mask
        loss = rms(E) + reg/2.0*(rms(P)+rms(Q))
        tqdm.write('epoch {}: {}'.format(epoch, loss))
        
        if loss < 0.001:
            break
            
    return P, Q


# In[ ]:


def SVD_batch(R, n_epochs=100, K=100, lr=0.005, reg=0.02,
             threshold=0.01):
    # U: the number of users
    # I: the number of items
    U, I = R.shape
    # P: user factor matrix
    P = np.random.rand(K, U)
    # Q: item factor matrix
    Q = np.random.rand(K, I)
    bU = np.zeros(U)
    bI = np.zeros(I)
    
    # only use rated elements of R to compute loss
    mask = np.ones(R.shape)
    mask[R == 0] = 0
    
    pre_loss = None
    chance = 2
    
    vld = Validater()
    # learning
    for epoch in trange(n_epochs, desc='epoch'):
        for u in trange(U):
            Pu = P[:,u] # (K, 1)
            # (1, I)
            Eu = (R[u,:] -Pu.T.dot(Q)
                  -bU[u] -bI)*mask[u,:] 
            # (K, 1)
            P[:,u] += lr*(np.mean(Eu*Q, axis=1)
                          - reg*Pu)
            bI += lr*(Eu.ravel() - reg*bI)
            
        for i in trange(I):
            Qi = Q[:,i] # (K, 1)
            # (U, 1)
            Ei = (R[:,i] -P.T.dot(Qi)
                 -bU -bI[i])*mask[:,i]
            # (K, 1)
            Q[:,i] += lr*(np.mean(Ei.T*P, axis=1)
                          - reg*Qi)
            bU += lr*(Ei.ravel() - reg*bU)
                    
        # compute & print loss
        E = (R - P.T.dot(Q)
             -bU.reshape(-1,1) -bI.reshape(1,-1))*mask
        loss = rms(E)# + reg/2.0*(rms(P)+rms(Q))
        tqdm.write('epoch {}'.format(epoch))
        tqdm.write(' train loss:{}'.format(loss))
        tqdm.write(' val loss:'+str(vld.validate(P, Q, bU, bI)))
        
        #if pre_loss is not None and pre_loss < loss:
        #    chance -= 1
        #    if chance < 0:
        #        break
        
        if loss < threshold:
            break
        
        pre_loss = deepcopy(loss)
        
    return P, Q, bU, bI


# In[ ]:


class Validater:

    # load test data
    test_file = './data/u1.test'
    df_test = pd.read_csv(test_file, delimiter='\t', header=None)
    df_test.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    
    def validate(self, P, Q, bU, bI):
        # predicted rating matrix
        df_test = self.df_test
        R_p = P.T.dot(Q) +bU.reshape(-1,1) +bI.reshape(1,-1)

        # get observations and predictions
        obs = df_test['rating'].values
        pred = R_p[df_test.values[:,0]-1, df_test.values[:,1]-1]

        # compute evaluation metrics
        results = {
            "MAE":mean_absolute_error(obs, pred),
            "RMSE":np.sqrt(mean_squared_error(obs, pred))
        }   

        return results


# In[ ]:


def main(fold_id=1):
    # load dataset
    data_file = './data/u{}.base'.format(fold_id)
    df_data = pd.read_csv(data_file, delimiter='\t', header=None)
    df_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    n_users = df_data.max()['user_id']
    n_items = df_data.max()['item_id']

    # change shape into user-item matrix
    ratings = df_data.pivot(index='user_id', columns='item_id',
                            values='rating').fillna(0)
    # fill the lack of no-rated item_id
    for item in range(n_items):
        item += 1
        if item not in ratings.columns:
            ratings.loc[:, item] = 0
            
    # learning
    R = ratings.values
    P, Q, bU, bI = SVD_batch(R, lr=0.3, n_epochs=10000)
    
    # save matrices
    np.savetxt('others/P.csv', P)
    np.savetxt('others/Q.csv', Q)
    np.savetxt('others/bU.csv', bU)
    np.savetxt('others/bI.csv', bI)


# In[ ]:


if __name__=='__main__':
    main()

