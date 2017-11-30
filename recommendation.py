#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import os
import argparse
import codecs

from tqdm import tqdm, trange
import joblib
from IPython.display import display, Image
import numpy as np
from numpy.linalg import solve
import pandas as pd
pd.set_option('display.max_colwidth', 1000)
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:

class MF_base:
    def predict(self, users, items):
        R_p = self.get_R_p
        pred = R_p[users, items]
        if len(users) == 1:
            return pred[0]
        return pred
    
    def get_R_p(self):
        X = self.X
        Y = self.Y
        R_p = X.dot(Y.T)
        return R_p
        
    def test(self, df_test):
        users = df_test['user_id']
        items = df_test['item_id']
        obs = df_test['rating']
        pred = self.predict(users, items)
        evaluation = np.sqrt(mean_squared_error(obs, pred))
        return evaluation
    
    def get_similarities(self):
        """
        Compute similartiteis matrix (m, m), using cosine
        """
        Y = self.Y
        sim = model.Y.dot(Y.T)
        norms = np.sqrt(np.diagonal(sim)).reshape(-1,1)
        return sim / norms/ norms.T
    
    def update(self):
        pass
        
    def fit(self, R ,df_val=None, out=None):
        """
        R (array): rating matrix
        df_test (DataFrame):
            - columns (user, item rating)
        """
        self.R = R
        
        f = self.f
        n_epochs = self.n_epochs
        
        m, n = R.shape
        X = np.random.rand(m, f) # user factors
        Y = np.random.rand(n, f) # item factors
        self.X = X
        self.Y = Y

        for epoch in trange(n_epochs):
            self.update()
                
            # compute train loss
            R_p = self.get_R_p()
            loss = np.sqrt(mean_squared_error(R[R!=0], R_p[R!=0]))
            tqdm.write('epoch {:03d}:'.format(epoch))
            tqdm.write('train loss: {}'.format(loss))
            
            # compute validation loss
            if df_val is not None:
                val_loss = self.test(df_val)
                tqdm.write('  val loss: {}'.format(val_loss))
            
        
        # save parameters
        params = {
            'R': R,
            'X': X,
            'Y': Y,
        }
        self.params = params
        if out is not None:
            if not os.path.exists(out):
                os.makedirs(out)
            
            params_file = os.path.join(out, 'parameters.pkl')
            joblib.dump(params, params_file, compress=True)
            
    def load_params(self, params_file):
        params = joblib.load(params_file)
        for k, v in params.items():
            exec('self.{} = v'.format(k))


# In[ ]:

class SVD_SGD(MF_base):
    def __init__(self, f=100, n_epochs=20, lr=0.005, reg=0.02, biased=False):
        self.f = f
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.biased = biased

    def update(self):
        lr = self.lr
        reg = self.reg
        R = self.R
        X = self.X
        Y = self.Y
        r_ui_notzero = zip(*np.where(R!=0))
        for u, i in r_ui_notzero:
            e = R[u,i] - X[u].dot(Y[i].T)
            X[u] += lr * (e*Y[i] - reg*X[u])
            Y[i] += lr * (e*X[u] - reg*Y[u])


# In[ ]:

class SVD_ALS(MF_base):
    def __init__(self, f=100, n_epochs=20, reg=0.02, biased=False):
        self.f = f
        self.n_epochs = n_epochs
        self.reg = reg
        self.biased = biased

    def fit(self, R ,df_val=None):
        """
        R (array): rating matrix
        df_test (DataFrame):
            - columns (user, item rating)
        """
        
        f = self.f
        n_epochs = self.n_epochs
        reg = self.reg
        
        m, n = R.shape
        X = np.random.rand(m, f) # user factors
        Y = np.random.rand(n, f) # item factors

        for epoch in trange(n_epochs):
            regI = reg * np.eye(f)

            YtY = Y.T.dot(Y)
            for u in trange(m):
                X[u] = solve((YtY + regI), 
                            R[u, :].dot(Y))

            XtX = X.T.dot(X)
            for i in trange(n):
                Y[i] = solve((XtX + regI), 
                            R[:, i].dot(X))
            
            R_p = X.dot(Y.T)
            loss = np.sqrt(mean_squared_error(R[R!=0], R_p[R!=0]))
            tqdm.write('epoch {:03d}:'.format(epoch))
            tqdm.write('train loss: {}'.format(loss))
            
            self.R_p = R_p
            if df_val is not None:
                val_loss = self.test(df_val)
                tqdm.write('  val loss: {}'.format(val_loss))


# In[ ]:

if __name__=='__main__':
    # get parameters from terminal
    parser = argparse.ArgumentParser(description='Matrix Factorization')
    parser.add_argument('-f', '--fold_id', type=int, default=1,
                       help='id of fold to validate')
    parser.add_argument('-e', '--n_epochs', type=int, default=10,
                       help='the number of epochs to train')
    parser.add_argument('-o', '--out', type=str, default='results',
                       help='the path where the training results will\
                       be')
    args = parser.parse_args()
    fold_id = args.fold_id
    n_epochs = args.n_epochs
    out = args.out
    
    # load train data
    data_file = './data/u{}.base'.format(fold_id)
    df_data = pd.read_csv(data_file, delimiter='\t', header=None)
    df_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    n_users = df_data.max()['user_id']
    n_items = df_data.max()['item_id']
    
    # load validation data
    val_file = './data/u{}.test'.format(fold_id)
    df_val = pd.read_csv(val_file, delimiter='\t', header=None)
    df_val.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    # change the shape of train data into user-item matrix
    ratings = df_data.pivot(index='user_id', columns='item_id',
                            values='rating').fillna(0)
    # fill the lack of no-rated item_id
    for item in range(n_items):
        item += 1
        if item not in ratings.columns:
            ratings.loc[:, item] = 0
            
    # learning
    model = SVD_SGD(n_epochs=n_epochs)
    R = ratings.values
    model.fit(R, df_val, out=out)


# In[ ]:

def recommend_topk(R, u, k=None, mask=None):
    with codecs.open('data/u.item', 'r', 'utf-8', 'ignore') as f:
        df_items = pd.read_csv(f, delimiter='|', header=None)
    
    rec = pd.DataFrame({
        'rating':R[u],
        'title':df_items[1],
    })
    
    if mask is None:
        mask = (R[u]!=0)
    rec = rec[mask]
    
    rec = rec.sort_values('rating', ascending=False)
    
    return rec[:k]


# In[ ]:

model = SVD_SGD()
model.load_params('results/test/parameters.pkl')

#recommend_topk(R=model.get_R_p(), u=0)
user = 0
rated = recommend_topk(R=model.R, u=user, k=10)
mask_unrated = (model.R[user]==0)
unrated = recommend_topk(R=model.get_R_p(), u=user, k=10, mask=mask_unrated)
rated


# In[ ]:

unrated


# In[ ]:



