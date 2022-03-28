import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from tqdm.auto import tqdm

from typing import Tuple, List, Optional

import numbers
import math
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler
torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_dtype(torch.float)

from torch import Tensor
# import custom_lstms
import torch.jit as jit

# np.random.seed(1234)
# torch.manual_seed(42)

from config import config

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_cells, output_dim):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.activation = nn.SiLU()
        
        self.layers.append(nn.Linear(input_dim,hidden_cells[0]))
        nn.init.kaiming_normal_(self.layers[-1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.layers[-1].bias, 0)
        for i in range(len(hidden_cells)-1):
            self.layers.append(nn.Linear(hidden_cells[i],hidden_cells[i+1]))
            nn.init.kaiming_normal_(self.layers[-1].weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(self.layers[-1].bias, 0)
            
        self.output_layer = nn.Linear(hidden_cells[-1], output_dim)
        nn.init.xavier_normal_(self.output_layer.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.constant_(self.output_layer.bias, 0)
        
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
            input = self.activation(input)
        output = self.output_layer(input)
        return output
        

class Network(nn.Module):
    def __init__(self, network, train_size, xdim, norm_func=lambda input, device: input,
                 inv_norm_func=lambda input, device: input, init_available=True, device=None, 
                 tf_prop=1., integrator='RK4', add_par_num=0):
        super(Network, self).__init__()
        
        if torch.cuda.is_available() and device is None:
            # torch.cuda.manual_seed(42)
            # torch.backends.cudnn.deterministic = True
            self.device = 'cuda'
        elif not torch.cuda.is_available() and device is None:
            self.device = 'cpu'
        else:
            self.device = device  
            
        self.net = network.to(self.device)
        self.tf_prop = torch.tensor(tf_prop).float()
        
        self.initial_x = nn.Parameter(torch.zeros(train_size, xdim).to(self.device) + 0.5, requires_grad = not init_available)
        self.additional_pars = nn.Parameter(torch.zeros(add_par_num).to(self.device) + 0.5, requires_grad = True)
        
        self.norm_func = lambda input: norm_func(input, self.device)
        self.inv_norm_func = lambda input: inv_norm_func(input, self.device)
        
        if integrator == 'Euler' or integrator == 'euler':
            self.integrator = self.Euler
        elif integrator == 'RK2' or integrator == 'rk2':
            self.integrator = self.RK2
        elif integrator == 'RK4' or integrator == 'rk4':
            self.integrator = self.RK4
        else: raise ValueError('Invalid integrator type')
    
    def forward(self, x: Tensor,  par: Tensor, time: Tensor, index: Optional[Tensor]) -> Tensor:
        """Forward pass
        Inputs:
        x of shape (n x T x dx). At time points T where data is not available, the item is np.nan
        par of shape (n x T x dp). Parameters must ALWAYS be available, otherwise a ValueError will be thrown.
        time of shape (n x T)
        index of shape (n,)
        """
        
        if torch.any(torch.isnan(par)):
            raise ValueError("Invalid parameter values found")
        
        dt = (time[:,1:,:] - time[:,:-1,:])
        # print(dt[:,:,:])
        # assert False

        x_arr = x.unbind(1)
        par_arr = par.unbind(1)
        dt_arr = dt.unbind(1)
        
        # Initialize output array
        x_out = []
        if index is not None:
            x_out.append(torch.where(torch.isnan(x_arr[0]), # Where condition
                                     self.inv_norm_func(self.initial_x[index,:]), # True clause
                                     x_arr[0])) # False clause
        else:
            x_out.append(x_arr[0])
#         print(x_out)
#         assert False
        
        switch = int(len(x_arr) * self.tf_prop)
        for i, (x_input_temp, p_input, dt_input) in enumerate(zip(x_arr, par_arr, dt_arr)):
            if i < switch: # Teacher Forcing:
                x_input = torch.where(torch.isnan(x_input_temp), # Where condition
                                     x_out[i], # True clause
                                     x_input_temp) # False clause
                # assert False, 'Teacher Forcing!'
#                 if i == 0:
#                     print(x_input)
#                     print('========================')
            else:
                x_input = x_out[i]
            
            if index is None: ind = 1
            else: ind = 0
            
            out = self.integrator(x_input, p_input, dt_input)
            x_out.append(out)
            
        #     print('Im forward')
        #     print(x_input_temp)
        #     print(x_input)
        #     print(out)
        # assert False

#             if (i==0):
#                 print('==============================')
#                 print('Outs')
#                 print(i)
#                 print(x_input)
#                 print(p_input)
#                 print(dt_input)
#                 print(out)
#                 print('==============================')


        x_outs = torch.stack(x_out[:],dim=1)
        return x_outs
    
    def output(self, x, par):
        
        ANN_input = torch.cat((self.norm_func(x), par), dim=-1)
        out = self.inv_norm_func(self.net(ANN_input))     


        return out
    
    def raw_output(self, x, par):
        ANN_input = torch.cat((self.norm_func(x), par), dim=-1)
        out = self.inv_norm_func(self.net(ANN_input))   
        return out

    def Euler(self, x, par, dt):
        
        ANN_output = self.output(x, par)
        x_out = x + ANN_output * dt
        
        return x_out
    
    def RK2(self, x, par, dt):
        ANN_input = x
        k1 = self.output(ANN_input, par)

        ANN_input = x + k1 * dt
        k2 = self.output(ANN_input, par)
        
        x_out = x + (dt/2) * (k1 + k2)
        return x_out
   
    def RK4(self, x, par, dt):
        
        ANN_input = x

        k1 = self.output(x, par)

        ANN_input = x + k1 * dt / 2
        k2 = self.output(ANN_input, par)
        
        ANN_input = x + k2 * dt / 2
        k3 = self.output(ANN_input, par)
        
        ANN_input = x + k3 * dt
        k4 = self.output(ANN_input, par)
        
        x_out = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        return x_out

class my_Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(my_Loss, self).__init__()
        self.reduction = reduction
        
    def forward(self, input, target):
        count = torch.sum((~torch.isnan(input))*1,dim=1)
#         print('========================')
#         print('count')
#         print(count)
#         print('input')
#         print(input[0,:20,0])
#         print('target')
#         print(target[0,:20,0])

        output = (input - target) ** 2
#         print('output')
#         print(output[0,:20,0])
#         print('output nansum')
#         print(torch.nansum(output,dim=1) / count)
#         print('final output')
#         print(torch.mean(torch.nansum(output,dim=1) / count))
        
        if self.reduction == 'mean':
            return torch.mean(torch.nansum(output,dim=1) / count)
        elif self.reduction == 'sum':
            return torch.sum(output)
        else:
            assert False, 'invalid reduction'

class Model_Train():
    def __init__(self, dataloader_train, dataloader_val, network, learning_rate=config["TRAINING"]["LEARNING_RATE"], device=None):
        if torch.cuda.is_available() and device is None:
            # torch.cuda.manual_seed(42)
            # torch.backends.cudnn.deterministic = True
            self.device = 'cuda'
        elif not torch.cuda.is_available() and device is None:
            self.device = 'cpu'
        else:
            self.device = device

        print('Using:', self.device)
        

        self.net = network.to(self.device)
        self.norm_func = network.norm_func

        self.var_factor = torch.tensor([1,10,1,1,1,1]).to(self.device)

        self.trainable_parameters = \
            sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('Trainable parameters: '+str(self.trainable_parameters))

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        
        self.lr = learning_rate

        initial_par_list = ['initial_x']
        other_par_list = ['additional_pars']
        init_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in initial_par_list, self.net.named_parameters()))))
        other_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in other_par_list, self.net.named_parameters()))))
        ANN_params = list(map(lambda x: x[1],list(filter(lambda kv: (kv[0] not in initial_par_list)\
                                            and (kv[0] not in other_par_list), self.net.named_parameters()))))
        
        print('MY LEARNING RATE IS ' + str(self.lr))
#         print(init_params)
#         print(other_params)
#         assert False
        
        self.optimizer = torch.optim.AdamW([
        {'params': ANN_params},
        {'params': init_params, 'lr': self.lr},
        {'params': other_params, 'lr': self.lr/5}],
            lr=self.lr, amsgrad=True, weight_decay=0.01)

#         self.criterion = my_Loss(reduction='mean')
        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)

        self.train_loss = []
        self.val_loss = []
        
        self.lr_epoch = []
        self.lr_track = []

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=100, factor=0.5, min_lr=0.000001, cooldown=110)

        self.epoch_shift = 50000000

        self.total_steps_OneCycle = int(np.ceil(config["DATA"]["N_TRAIN"]/config["TRAINING"]["BATCH_SIZE"]) \
                               * (config["TRAINING"]["EPOCHS"]-self.epoch_shift))
        
    def train(self, epoch):
        """Train model."""
        self.net.train()
        cnt, sum_loss = 0, 0
        iters = len(self.dataloader_train)
        
        # Switch to OneCycleLR LR rate scheduler
        if epoch == self.epoch_shift:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=[self.lr,self.lr, self.lr], total_steps=self.total_steps_OneCycle, final_div_factor=1e2,
                                                            )
            # self.net.tf_prop = torch.tensor(0.2).float()
            # print('Shifting to Autoregressive at epoch: ' + str(epoch))
 
        for (x, p, t, index) in self.dataloader_train:
            self.optimizer.zero_grad()
            x_in = x[0][:,:-1,:]
            
            xout_hat = self.net(x_in.to(self.device), p[0].to(self.device), t[0].to(self.device), index)
            
            # Normalize vectors before loss calculation
            x_out_norm = self.norm_func(x[0].to(self.device)) * self.var_factor
            x_out_hat_norm = self.norm_func(xout_hat) * self.var_factor

            # # Weigh the first available datapoint higher
            # fac = 10
            # for i in range(x_out_norm.shape[0]):
            #     for j in range(x_out_norm.shape[2]):
            #         x_out_norm[i,:,j][~torch.isnan(x_out_norm)[i,:,j]][:5] = \
            #             x_out_norm[i,:,j][~torch.isnan(x_out_norm)[i,:,j]][:5] * fac
            #         x_out_hat_norm[i,:,j][~torch.isnan(x_out_norm)[i,:,j]][:5] = \
            #             x_out_hat_norm[i,:,j][~torch.isnan(x_out_norm)[i,:,j]][:5] * fac

            
            # print('I\'m time')
            # if epoch > 0:
            time_var = (4 * torch.exp(-t[0]/10) + 1).to(self.device)
            x_out_norm = x_out_norm * time_var
            x_out_hat_norm = x_out_hat_norm * time_var

            # print('Im model')
            # print(x_out_norm.shape)
            # print(x_out_norm[~torch.isnan(x_out_norm)].shape)
            # print(x_out_hat_norm.shape)
            # print(x_out_hat_norm[~torch.isnan(x_out_norm)].shape)
            # assert False

            # print('training time shape')
            # print('preds')
            # print(xout_hat[:,:2,:])
            # print(x_out_hat_norm[:,:2,:])
            # print('trues')
            # print(x[0].to(self.device)[:,:2,:])
            # print(x_out_norm[:,:2,:])
            # assert False

            # Calculate loss

            loss = self.criterion(x_out_norm[~torch.isnan(x_out_norm)],
                             x_out_hat_norm[~torch.isnan(x_out_norm)])
            loss.backward()
            # assert False
            
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)

            # if epoch < 500:
            #     torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
            # else:
            #     torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
            
            self.optimizer.step()
            
            # Use for 1-Cycle
            if epoch >= self.epoch_shift:
                self.scheduler.step()
                self.lr_epoch.append(epoch + cnt / iters)
                self.lr_track.append(self.scheduler.get_last_lr())
            
            sum_loss += loss.detach().cpu().numpy()
            cnt += 1
        
#         Use for lr reduce on plateau
        if epoch < self.epoch_shift and epoch > 0:
            self.scheduler.step(sum_loss / cnt)
            self.lr_epoch.append(epoch)
            self.lr_track.append(self.scheduler._last_lr)
        
        self.train_loss.append(sum_loss/cnt)

        return sum_loss/cnt
    
    def validate(self, epoch):
        """Validate model."""
        self.net.eval()
        cnt, sum_loss = 0, 0
        with torch.no_grad():
            for (x, p, t, index) in self.dataloader_val:
                x_in = x[0][:,:-1,:]

                xout_hat = self.net(x_in.to(self.device), p[0].to(self.device), t[0].to(self.device), index=None)

                # Normalize vectors before loss calculation
                x_out_norm = self.norm_func(x[0].to(self.device)) * self.var_factor
                x_out_hat_norm = self.norm_func(xout_hat) * self.var_factor

                time_var = (4 * torch.exp(-t[0]/10) + 1).to(self.device)
                x_out_norm = x_out_norm * time_var
                x_out_hat_norm = x_out_hat_norm * time_var


                # Calculate loss
                loss = self.criterion(x_out_norm[~torch.isnan(x_out_norm)],
                                 x_out_hat_norm[~torch.isnan(x_out_norm)])

                sum_loss += loss.detach().cpu().numpy()
                cnt += 1
        self.val_loss.append(sum_loss/cnt)
        return sum_loss/cnt


    def save_network(self, name):
        """Save network weights and training loss history."""
        filename = name +'.net'
        torch.save(self.net.state_dict(), filename)
        np.save(name+'_training_loss.npy', np.array(self.train_loss))
        np.save(name+'_validation_loss.npy', np.array(self.val_loss))
        np.save(name+'_lr_epoch.npy', np.array(self.lr_epoch))
        np.save(name+'_lr_track.npy', np.array(self.lr_track))
        return name

    def load_network(self, name):
        """Load network weights and training loss history."""
        filename = name + '.net'
        self.net.load_state_dict(torch.load(filename))
        self.train_loss = np.load(name+'_training_loss.npy').tolist()
        self.val_loss = np.load(name+'_validation_loss.npy').tolist()

def progress(train_loss, val_loss):
    """Define progress bar description."""
    return "Train/Loss: {:.2e}  Val/Loss: {:.2e}".format(
        train_loss, val_loss)