
# coding: utf-8

# In[ ]:


from copy import deepcopy
import os
import codecs

from IPython.display import display
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
import numpy as np
import pandas as pd


# In[ ]:


data_file = './data/u1.base'
df_data = pd.read_csv(data_file, delimiter='\t', header=None)
df_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
df_data


# In[ ]:


data_file = './data/u.item'
genre = [l.rstrip().split('|')[0] for l in open('data/u.genre')][:-1]

# for UnicodeDecodeError
with codecs.open(data_file, 'r', 'utf-8', 'ignore') as f:
    df_item = pd.read_table(f, delimiter='|', header=None)
    df_item.columns = [
        'movie_id', 
        'title', 
        'release_date', 
        'video_release_date',
        'IMDb_URL',
    ] + genre
df_item


# In[ ]:


data_file = './data/u.user'
df_user = pd.read_csv(data_file, delimiter='|', header=None)
df_user.columns = ['user_id', 'age', 'gender', 'occupation', 'zip code']
df_user


# In[ ]:


n_users = 943
n_items = 1682
ratings = np.zeros((n_users, n_items))
for index, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
    user_id = row['user_id']
    item_id = row['item_id']
    rating = row['rating']
    ratings[user_id-1][item_id-1] = rating
ratings


# In[ ]:


ratings[1-1][1-1]


# In[ ]:


def SVD(R, k, n_steps=100,
        alpha=0.0002, beta=0.02, threshold=0.001):
    """
    R: rating matrix
    U: user factor matrix
    I: item factor matrixf
    alpha: learning rate
    beta: regularization parameter
    """
    U = np.random.rand(k, R.shape[0])
    I = np.random.rand(k, R.shape[1])
    # TODO 0(ratingなし) はmaskで抜かす
    mask = np.ones(R.shape)
    mask[R == 0] = 0
    display(mask)
    pre_loss = None
    epoch = 0
    mx = deepcopy(alpha)
    mn = 0
    n_cycle = 14
    first_t_max = 1
    t_mult = 2
    ls_t_max = [first_t_max * t_mult**i for i in range(n_cycle)]
    #n_steps_c = int(np.floor(n_steps/n_cycles))
    #ts = [ i for i in range(n_steps_c,n_steps+1,n_steps_c)]
    
    for t_max in ls_t_max:
        for t_cur in trange(t_max):
            err = R - (U.T.dot(I))*mask
            loss = np.mean((err*err).flatten())                     + beta/2.0 * (np.linalg.norm(U) + np.linalg.norm(I))
            print(epoch, loss)
            U *= (1-alpha*beta)
            U += np.sum(2*alpha*err.T[np.newaxis, :,:]*I[:,:,np.newaxis], axis=1)
            I *= (1-alpha*beta)
            I += np.sum(2*alpha*  err[np.newaxis, :,:]*U[:,:,np.newaxis], axis=1)

            #if pre_loss is not None:
            #    sub = pre_loss - loss
            #    if sub < loss*1e-4:
            #        print('pre_loss - loss:', sub)
            #        print('alpha is dived by 10')
            #        alpha /= 10
            alpha = mn + (mx-mn)*(1+np.cos(t_cur/t_max*np.pi))/2
            print('lr:', alpha)
            if epoch % 500 == 0 and epoch != 0:
                print('10 times')
                alpha *= 10

            pre_loss = deepcopy(loss)
            epoch += 1

            if loss < threshold or epoch>n_steps:
                break

            print()
    return U, I


# In[ ]:


U, I = SVD(R=ratings, n_steps=20000, k=2, alpha=4e-5, beta=0, threshold=0.01)
ratings_pred = U.T.dot(I)


# In[ ]:


from itertools import product
for u, i in product(range(10), range(10)):
    print(ratings[u][i], ratings_pred[u][i])

