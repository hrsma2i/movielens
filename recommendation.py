#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import os
import argparse
import codecs
import json

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
        R_p = self.get_R_p()
        pred = R_p[users, items]
        if len(users) == 1:
            return pred[0]
        return pred
    
    def get_R_p(self):
        X = self.X
        Y = self.Y
        R_p = X.dot(Y.T)
        if self.biased:
            bu = self.bu
            bi = self.bi
            b = self.b
            R_p += b + bu.reshape(-1,1) + bi.reshape(1,-1)
        return R_p
        
    def test(self, df_test):
        users = df_test['user_id'] - 1
        items = df_test['item_id'] - 1
        obs = df_test['rating']
        pred = self.predict(users, items)
        evaluation = np.sqrt(mean_squared_error(obs, pred))
        return evaluation
    
    def update(self):
        pass
        
    def fit(self, R ,df_val=None, out=None):
        """
        R (array): rating matrix
        df_val (DataFrame):
            - columns (user, item rating)
        """
        self.R = R
        
        f = self.f
        n_epochs = self.n_epochs
        biased = self.biased
        
        m, n = R.shape
        self.m = m
        self.n = n
        if biased:
            bu = np.zeros(m)
            bi = np.zeros(n)
            b = np.mean(R[R!=0])
            self.bu = bu
            self.bi = bi
            self.b  = b
        X = np.random.rand(m, f) # user factors
        Y = np.random.rand(n, f) # item factors
        self.X = X
        self.Y = Y
        
        logs = []
        if not os.path.exists(out):
            os.makedirs(out)
        
        for epoch in trange(n_epochs):
            self.update()
                
            # compute train loss
            R_p = self.get_R_p()
            loss = np.sqrt(mean_squared_error(R[R!=0], R_p[R!=0]))
            log = {'loss':loss}
            tqdm.write('epoch {:03d}:'.format(epoch))
            tqdm.write('train loss: {}'.format(loss))
            
            # compute validation loss
            if df_val is not None:
                val_loss = self.test(df_val)
                log['val_loss'] = val_loss
                tqdm.write('  val loss: {}'.format(val_loss))
            
            # dump log
            logs.append(log)
            with open(os.path.join(out,'log'), 'w') as f:
                json.dump(logs, f, indent=4)
            
        
        # save parameters
        params = {
            'R': R,
            'X': X,
            'Y': Y,
        }
        if biased:
            params['bu'] = bu
            params['bi'] = bi
            params['b']  = b
        self.params = params
            
        params_file = os.path.join(out, 'parameters.pkl')
        joblib.dump(params, params_file, compress=True)
            
    def load_params(self, params_file):
        params = joblib.load(params_file)
        for k, v in params.items():
            exec('self.{} = v'.format(k))


# In[ ]:

class SVD_SGD(MF_base):
    def __init__(self, f=100, n_epochs=20, lr=0.001, reg=0.02, biased=False):
        self.f = f
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.biased = biased

    def update(self):
        lr = self.lr
        reg = self.reg
        biased = self.biased
        R = self.R
        X = self.X
        Y = self.Y
        if biased:
            bu = self.bu
            bi = self.bi
            b = self.b
        
        r_ui_notzero = zip(*np.where(R!=0))
        for u, i in r_ui_notzero:
            e = R[u,i] - X[u].dot(Y[i].T)
            if biased:
                e -= (b + bu[u] + bi[i])
                bu[u] += lr * (e - reg*bu[u])
                bi[i] += lr * (e - reg*bi[i])
            X[u] += lr * (e*Y[i] - reg*X[u])
            Y[i] += lr * (e*X[u] - reg*Y[u])


# In[ ]:

class SVD_ALS(MF_base):
    def __init__(self, f=100, n_epochs=20, lr=0.001, reg=0.02, biased=False):
        self.f = f
        self.n_epochs = n_epochs
        self.reg = reg
        self.biased = biased

    def update(self):
        reg = self.reg
        biased = self.biased
        R = self.R
        X = self.X
        Y = self.Y
        f = self.f
        m = self.m
        n = self.n
        if biased:
            bu = self.bu
            bi = self.bi
            b = self.b
        
        if biased:
            regI = reg * np.eye(f+1)

            Y1 = np.concatenate((Y, np.ones((n,1))), axis=1)
            Y1tY1 = Y1.T.dot(Y1)
            for u in trange(m):
                Xb = solve((Y1tY1 + regI), 
                            (R[u, :]-bi-b).dot(Y1))
                X[u], bu[u] = Xb[:-1], Xb[-1]

            X1 = np.concatenate((X, np.ones((m,1))), axis=1)
            X1tX1 = X1.T.dot(X1)
            for i in trange(n):
                Yb = solve((X1tX1 + regI), 
                            (R[:, i]-bu-b).dot(X1))
                Y[i], bi[i] = Yb[:-1], Yb[-1]
        else:
            regI = reg * np.eye(f)

            YtY = Y.T.dot(Y)
            for u in trange(m):
                X[u] = solve((YtY + regI), 
                            R[u, :].dot(Y))

            XtX = X.T.dot(X)
            for i in trange(n):
                Y[i] = solve((XtX + regI), 
                            R[:, i].dot(X))


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

if __name__=='__main__':
    # get parameters from terminal
    parser = argparse.ArgumentParser(description='Matrix Factorization')
    parser.add_argument('-i', '--fold_id', type=int, default=1,
                       help='id of fold to validate')
    parser.add_argument('-o', '--out', type=str, default='results',
                       help='the path where the training results will\
                       be')
    parser.add_argument('-e', '--n_epochs', type=int, default=200,
                       help='the number of epochs to train')
    parser.add_argument('-f', '--n_factor', type=int, default=80,
                       help='the number of factors')
    parser.add_argument('-b', '--biased', action='store_true',
                       help='biased or not')
    parser.add_argument('-l', '--lr', type=float, default=0.001,
                       help='learning rate')
    parser.add_argument('-r', '--reg', type=float, default=0.01,
                       help='regularization parameter')
    parser.add_argument('--opt', type=str, default='sgd',
                       help='optimization method')
    args = parser.parse_args()
    fold_id = args.fold_id
    n_epochs = args.n_epochs
    out = args.out
    biased = args.biased
    reg = args.reg
    lr = args.lr
    opt = args.opt
    
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
    print(opt)
    if opt == 'sgd':
        model = SVD_SGD(n_epochs=n_epochs, biased=biased,
                       lr=lr, reg=reg)
    elif opt == 'als':
        model = SVD_ALS(n_epochs=n_epochs, biased=biased,
                       lr=lr, reg=reg)
    R = ratings.values
    model.fit(R, df_val, out=out)

