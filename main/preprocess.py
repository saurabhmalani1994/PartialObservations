import sys
sys.path.append("..")

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm
from config import config
import torch
import random

np.random.seed(1234)
torch.manual_seed(42)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, x_input, par_input, max_dt=config["DATA"]["MAX_DELTA_T"], duplicate_reverse=config["DATA"]["DUP_REVERSE"]):
        """INPUTS: 
        x_input of shape (n x dx). Each element of this object array is a tuple (x, t). 
        x and t are numpy arrays of shape (Tx,). 
        x is array of observations of the variable, and t the corresponding times for each of the observations. 
        
        par_input of shape(n x dp). Each element of this object array is a tuple (p,t).
        p and t are numpy arrays of shape (Tp,). 
        p is array of time steps of parameter change, and t the corresponding times.
        Parameter values are assumed to be constant until next specified update time
        
        max_dt hyper parameter that determines the largest step size the templated numerical integrator should take
        
        OUTPUTS:
        x_arr_out of shape (n x T x dx). At time points T where data is not available, the item is np.nan
        par_arr_out of shape (n x T x dp). Parameters must ALWAYS be available, otherwise a ValueError will be thrown.
        full_times_arr of shape (n,), with each element of shape (T,)
        ids of shape (n,)
        """
        
        n_train = x_input.shape[0]
        x_dim = x_input.shape[1]
        p_dim = par_input.shape[1]
        
        self.ids = np.arange(n_train)
        self.full_times_arr = []

        if duplicate_reverse:
            full_times_arr_dup = []

        max_t_length = 0
        for i in range(n_train):
            # Collate time vectors and concatenate
            t = []
            t_first_x = np.inf
            for j in range(x_dim):
                t.append(x_input[i,j][1])
                t_first_x = min(t_first_x, min(x_input[i,j][1]))
            for j in range(p_dim):
                t.append(par_input[i,j][1])
                # Ensure all parameters are specified from the earliest time point available
                if min(par_input[i,j][1]).round(decimals=5) > t_first_x.round(decimals=5):
                    raise ValueError("Parameter number " + str(j+1) + " not specified at earliest time")
            full_times = np.sort(np.unique(np.concatenate(t).flatten())).round(decimals=5)
            # if duplicate_reverse:
            #     full_times_dup = np.copy(np.flip(full_times))
            #     full_times_arr_dup.append(np.copy(np.flip(self.insert_intermediate(full_times_dup, -max_dt))))
            
            # Insert intermediate time points for solver stepping
            self.full_times_arr.append(self.insert_intermediate(full_times, max_dt))
            max_t_length = max(max_t_length, self.full_times_arr[i].size)
        
        # Insert padding at end of shorter time vectors to make them all the same length
        for i in range(n_train):
            self.full_times_arr[i] = np.append(self.full_times_arr[i], 
                            np.zeros(max_t_length - self.full_times_arr[i].size) + max(self.full_times_arr[i]) + max_dt)
        
        self.x = np.empty((n_train, max_t_length, x_dim))
        self.x[:] = np.nan
        
        self.p = np.empty((n_train, max_t_length, p_dim))
        self.p[:] = np.nan
        
        for i in range(n_train):
            for j in range(x_dim):
                self.x[i,np.in1d(self.full_times_arr[i],x_input[i,j][1].round(decimals=5)),j] = x_input[i,j][0]
            for j in range(p_dim):
                self.p[i,:,j] = par_input[i,j][0][-1]
                for k in range(par_input[i,j][1].size - 1):
                    self.p[i,j,self.full_times_arr[i]<par_input[i,j][1][-1-k]] = par_input[i,j][0][-2-k]

        # if duplicate_reverse:
        #     self.ids = np.arange(n_train*2)
        #     x_dup = np.empty((n_train, max_t_length, x_dim))
        #     p_dup = np.empty((n_train, max_t_length, p_dim))
        #     t_dup = []
        #     for i in range(n_train):
        #         for j in range(x_dim):
        #             x_dup[i,:,j] = np.flip(self.x[i,:,j])
        #         for j in range(p_dim):
        #             p_dup[i,:,j] = np.flip(self.p[i,:,j])
        #         self.full_times_arr.append(full_times_arr_dup[i])
        #     self.x = np.concatenate((self.x, x_dup), axis=0)
        #     self.p = np.concatenate((self.p, p_dup), axis=0)
                

            
    def insert_intermediate(self, t_samp, max_dt):
        t_arr = []
        t_arr.append(t_samp[0])
        for i in range(len(t_samp)-1):
            t_step = np.minimum(t_samp[i+1] - t_samp[i], max_dt)
            t_current = t_arr[-1]
            t_add = []
            while t_current < t_samp[i+1]:
                dt = -1
                while dt < 0:
                    # dt = t_step * (1 - np.random.gamma(shape=1, scale=0.1))
                    dt = max_dt
                t_current = t_current + dt
                t_add.append(dt)
            # random.shuffle(t_add)
            t_current = t_arr[-1]
            for item in t_add:
                t_current = t_current + item
                t_arr.append(t_current)
            t_arr[-1] = t_samp[i+1]
            # t_arr.append(np.arange(start=t_samp[i],stop=t_samp[i+1],step=max_dt))
        # t_arr.append(np.array([t_samp[-1]]))
        # t_arr[-1] = t_samp[i+1]
        return np.sort(np.unique(np.array(t_arr).round(decimals=5).flatten()))
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        
        x = torch.tensor(self.x[self.ids[index],:,:]).float(),
        p = torch.tensor(self.p[self.ids[index],:,:]).float(),
        t = torch.tensor(self.full_times_arr[self.ids[index]]).float().unsqueeze(-1),

        
        return x, p, t, index   