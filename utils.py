import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from tqdm.auto import tqdm

from typing import Tuple, List, Optional

import numbers
import math
import torch
import torch.nn as nn
# torch.use_deterministic_algorithms(True)

from torch.optim.lr_scheduler import _LRScheduler
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.double)

from torch import Tensor
# import custom_lstms
import torch.jit as jit

config = {}
config["DATA"] = {}
config["DATA"]["TMAX"] = 10
config["DATA"]["L_TRAJECTORIES"] = 200
config["DATA"]["N_TRAIN"] = 200
config["DATA"]["N_VAL"] = 50
config["DATA"]["N_TEST"] = 1
config["DATA"]["PATH"] = 'data/'
config["DATA"]["SKIP"] = 20
config["DATA"]["Y_SAMPLE"] = 20

config["PAR"] = {}
config["PAR"]["Da"] = 0.33
config["PAR"]["B"] = 11
config["PAR"]["beta"] = 3

config["TRAINING"] = {}
config["TRAINING"]["BATCH_SIZE"] = 256
config["TRAINING"]["LEARNING_RATE"] = 5e-3
config["TRAINING"]["EPOCHS"] = 2000 # 1000 # 1259 # 2539 # 5260

config["MODEL"] = {}
config["MODEL"]["NUM_HIDDEN"] = [32, 32]

np.random.seed(1234)
torch.manual_seed(42)
# torch.manual_seed(1234)


def cstr_initial_conditions(ic='random'):
    if ic == 'random':
        return (0.4 + 0.6*np.random.random(), 0.5 + 4*np.random.random())
    if ic == 'saurabh':
        x10 = 0.82
        x20 = 4.2
        return (x10, x20)


def f_cstr(t, y, Da, beta, B):
    """Time derivatives of the CSTR model."""
    x1, x2 = y
    dx1dt = -x1 + Da * (1-x1) * np.exp(x2)
    dx2dt = -x2 + B * Da * (1-x1) * np.exp(x2) - beta * x2

    return [dx1dt, dx2dt]

# def Jac_cstr(t, y, Da, beta, B):
#     """Time derivatives of the CSTR model."""
#     x1, x2 = y

#     dx1dotdx1 = -1 - Da * np.exp(x2)
#     dx1dotdx2 = Da * (1-x1) * np.exp(x2)
    
#     dx2dotdx1 = - B * Da * np.exp(x2)
#     dx2dotdx2 = -1 - beta + B * Da * (1-x1) * np.exp(x2)

#     return [[dx1dotdx1, dx1dotdx2], [dx2dotdx1, dx2dotdx2]]


def get_pars(Da: np.float=0.085, B: np.float=22, beta: np.float=3):
#     B = 22.0
#     beta = 3.0
    

    pars = Da, beta, B
    return pars


def integrate_cstr(tmin=0, tmax=20, T=2000, y0=None, verbose=False, Da=0.085, B=22, beta=3):

    pars = get_pars(Da,B,beta)
    if y0 is None:
        y0 = cstr_initial_conditions()

    sol = solve_ivp(f_cstr, y0=y0, t_span=[0, tmax],
                    t_eval=np.linspace(tmin, tmax, T+1), args=pars,
                    rtol=1e-5, atol=1e-8)
    if verbose:
        print(sol.message)

    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(sol.t, sol.y[0], label='$X$')
        ax.plot(sol.t, sol.y[1], label=r'$\theta$')
        ax.set_xlabel('$t$')
        ax.set_ylabel('')
        # plt.savefig('')
        plt.show()
    return sol


class CSTRDataset(torch.utils.data.Dataset):
    """Dataset of transients obtained from a Brusselator."""

    def __init__(self, n_train, tmax, l_trajectories, Da=0.085, B=11, beta=3, random=False):
        self.ids = np.arange(n_train)
        self.x1 = []
        self.x1_out = []
        self.x1_data = []
        self.x2 = []
        self.x2_out = []
        skip = config["DATA"]["SKIP"]
        y_sample = config["DATA"]["Y_SAMPLE"]
        
        if n_train == 1:
            self.Da = np.array([Da])
        else:
            if random:
                self.Da = np.random.uniform(0.2,0.5,n_train)   
#                 self.Da = np.concatenate((np.random.uniform(0.2,0.25,int(n_train/2)),np.random.uniform(0.45,0.5,int(n_train/2)))) 
#                 self.Da = np.random.uniform(0.33,0.33,n_train)   
            else:
#                 self.Da = np.linspace(0.2,0.5,n_train)
                self.Da = np.concatenate((np.linspace(0.2,0.25,int(n_train/2)),np.linspace(0.25,0.5,int(n_train/2))))
#                 self.Da = np.linspace(0.33,0.33,n_train)   
        
        count = 0
        
        for _ in tqdm(range(n_train), leave=True, position=0):
#             index = int(np.floor(count/10))
            index = count
            
            sol = integrate_cstr(tmax=tmax, T=l_trajectories, Da=self.Da[index], B=B, beta=beta)
            self.x2.append(sol.y[1, skip:-1])  # Observed variable at t shape (N, T)
            self.x2_out.append(sol.y[1, skip+1:])  # Obeserved variabel at t+1
            
            x1 = np.zeros(sol.y[0, skip:].size) - 1
            x1[::y_sample] = sol.y[0, skip::y_sample]
            
            self.x1.append(x1[:-1])  # Unobserved variable at t
            self.x1_out.append(x1[1:])  # Unobserved variable at t
            
            self.x1_data.append(sol.y[0, skip:].copy())
            
            count += 1
        self.tt = sol.t[skip:] - sol.t[skip]  # Time array
        self.delta_t = self.tt[1]-self.tt[0]  # delta t
        
        print('Using dt of '+str(self.delta_t))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x1 = self.x1[self.ids[index]]
        x1_out = self.x1_out[self.ids[index]]
        x2 = self.x2[self.ids[index]]
        x2_out = self.x2_out[self.ids[index]]
        delta_t = self.delta_t
        Da = self.Da[self.ids[index]]
        return torch.tensor(x1, dtype=torch.float64), \
            torch.tensor(x1_out, dtype=torch.float64), \
            torch.tensor(x2, dtype=torch.float64), \
            torch.tensor(x2_out, dtype=torch.float64), \
            torch.tensor(delta_t, dtype=torch.float64).unsqueeze(-1), \
            torch.tensor(Da, dtype=torch.float64).unsqueeze(-1), \

    def save_data(self, path, filename):
        np.savez(path+filename, x=self.x, y=self.y, y_data=self.y_data, tt=self.tt, ids=self.ids)

class Snake(nn.Module):
    """Nonlinear activation function."""
    
    def __init__(self, alpha = None):
            '''
            Initialization.
            INPUT:
                - in_features: shape of the input
                - aplha: trainable parameter
                aplha is initialized with zero value by default
            '''
            super(Snake,self).__init__()

            # initialize alpha
            if alpha == None:
                self.alpha = nn.Parameter(torch.tensor(1.0)) # create a tensor out of alpha
            else:
                self.alpha = nn.Parameter(torch.tensor(alpha)) # create a tensor out of alpha

            self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, input_tensor):
        """Forward pass through activation function."""
        return input_tensor + (1/self.alpha) * (torch.sin(input_tensor * self.alpha)) ** 2
        
# class Network(nn.Module):
class Network(jit.ScriptModule):

    def __init__(self, hidden_cells, minmaxes, tau=torch.tensor([0.05]), beta=3, B=11, Da=0.33, device=None):
        super(Network, self).__init__()
        self.hidden_cells = hidden_cells
        
        if torch.cuda.is_available() and device is None:
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True
            self.device = 'cuda'
        elif not torch.cuda.is_available() and device is None:
            self.device = 'cpu'
        else:
            self.device = device        
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(3,hidden_cells[0]))
        for i in range(len(hidden_cells)-1):
            self.layers.append(nn.Linear(hidden_cells[i],hidden_cells[i+1]))
            
        self.output_layer = nn.Linear(hidden_cells[-1],2)

        self.x1min, self.x1max, self.x2min, self.x2max = minmaxes
    
        self.activation = Snake()
#         self.activation = nn.SiLU()
        
        self.sigmoid = nn.Sigmoid()

#         self.model = nn.Sequential(
#             nn.Linear(3,32),
#             self.activation(),
#             nn.Linear(32,32),
#             self.activation(),
# #             nn.Linear(16,16),
# #             self.activation(),
#             nn.Linear(32,2),
#         )
        
#         self.model = nn.Sequential(
#             nn.LSTM(2,10,batch_first=True),
#             nn.Linear(10,2),
#         )
        
        
#         self.tau = tau
#         self.beta = torch.tensor(beta)
#         self.B = torch.tensor(B)
#         self.Da = torch.tensor(Da)

    @jit.script_method
    def forward_bk2(self, x1: Tensor, x2: Tensor, dt: Tensor, Da: Tensor, warmup: int) -> Tuple[Tensor, Tensor]:
        """Forward pass"""
        
        x1_input = (x1[:,:warmup] - self.x1min) / (self.x1max - self.x1min)
        x2_input = (x2[:,:warmup] - self.x2min) / (self.x2max - self.x2min)
        Da_input = (Da.repeat(1,x1_input.size()[1]) - 0.2) / (0.5 - 0.2)
#         Da_input = Da.repeat(1,x1_input.size()[1])
        
#         ANN_input = torch.stack((x1_input,x2_input),dim=-1)
        ANN_input = torch.stack((x1_input,x2_input, Da_input),dim=-1)

#         ANN_output = self.model(ANN_input)
        for layer in self.layers:
            ANN_input = layer(ANN_input)
            ANN_input = self.activation(ANN_input)
        ANN_output = self.output_layer(ANN_input)
            
#         x1_outs = (self.sigmoid(ANN_output[:,:,0])*1.2-0.1) * (self.x1max - self.x1min) + self.x1min
#         x2_outs = (self.sigmoid(ANN_output[:,:,1])*1.2-0.1) * (self.x2max - self.x2min) + self.x2min
        x1_outs_warmup = x1[:,:warmup] + ANN_output[:,:,0]
        x2_outs_warmup = x2[:,:warmup] + ANN_output[:,:,1]
        
#         x1_outs_warmup = (self.sigmoid(ANN_output[:,:,0])*1.2-0.1) * (self.x1max - self.x1min) + self.x1min
#         x2_outs_warmup = (self.sigmoid(ANN_output[:,:,1])*1.2-0.1) * (self.x2max - self.x2min) + self.x2min
        
        x1_out = []
        x2_out = []
        
        if (x1.size()[1] > warmup):
            x1_out.append(x1[:,warmup])
            x2_out.append(x2[:,warmup])
            
        Da_input = (Da.squeeze(-1) - 0.2) / (0.5 - 0.2)
#         Da_input = Da.squeeze(-1)
        
        for j in range(x1.size()[1]-warmup):
            x1_input = (x1_out[j] - self.x1min) / (self.x1max - self.x1min)
            x2_input = (x2_out[j] - self.x2min) / (self.x2max - self.x2min)

            
#             ANN_input = torch.stack((x1_input,x2_input),dim=-1)
#             print(x1_input.shape)
#             print(x2_input.shape)
#             print(Da_input.shape)
#             print(Da_input.squeeze(-1).shape)
            ANN_input = torch.stack((x1_input,x2_input,Da_input),dim=-1)

#             ANN_output = self.model(ANN_input)
            
            for layer2 in self.layers:
                ANN_input = layer2(ANN_input)
                ANN_input = self.activation(ANN_input)
            ANN_output = self.output_layer(ANN_input)
            
            

            
            x1_integ =  ANN_output[:,0]  + x1_out[j] # * dt.squeeze()
            x2_integ =  ANN_output[:,1]  + x2_out[j] # * dt.squeeze()
#             x1_integ = (self.sigmoid(ANN_output[:,0])*1.2-0.1) * (self.x1max - self.x1min) + self.x1min
#             x2_integ = (self.sigmoid(ANN_output[:,1])*1.2-0.1) * (self.x2max - self.x2min) + self.x2min

#             x1_integ = self.sigmoid(ANN_output[:,0])*1.1
#             x2_integ = self.sigmoid(ANN_output[:,1])*10

            x1_integ = torch.clip(x1_integ, min = 0, max = self.x1max*1000)
            x2_integ = torch.clip(x2_integ, min = 0, max = self.x2max*1000)
    
            x1_out.append(x1_integ)
            x2_out.append(x2_integ)

            
        if (x1.size()[1] > warmup):
            x1_outs = torch.cat((x1_outs_warmup,torch.stack(x1_out[1:],dim=1)),dim=1)
            x2_outs = torch.cat((x2_outs_warmup,torch.stack(x2_out[1:],dim=1)),dim=1)
        else:
            x1_outs = x1_outs_warmup
            x2_outs = x2_outs_warmup
        
        return x1_outs, x2_outs
    
    def forward_bk(self, x1: Tensor, x2: Tensor, dt: Tensor, Da: Tensor, warmup: int) -> Tuple[Tensor, Tensor]:
        """Forward pass"""
        
        x1input = (x1 - self.x1min) / (self.x1max - self.x1min)
        x2input = (x2 - self.x2min) / (self.x2max - self.x2min)

        ANN_input = torch.stack((x1input,x2input),dim=-1)

        ANN_output = self.model(ANN_input)
            
            
#         x1_outs = (self.sigmoid(ANN_output[:,:,0])*1.2-0.1) * (self.x1max - self.x1min) + self.x1min
#         x2_outs = (self.sigmoid(ANN_output[:,:,1])*1.2-0.1) * (self.x2max - self.x2min) + self.x2min
        x1_outs = x1 + ANN_output[:,:,0]
        x2_outs = x2 + ANN_output[:,:,1]
        
        return x1_outs, x2_outs

    @jit.script_method
    def forward(self, x1: Tensor, x2: Tensor, dt: Tensor, Da: Tensor, warmup: int) -> Tuple[Tensor, Tensor]:
        """Forward pass"""
        
        
        
        x1_inputs = x1.unbind(1)
        x2_inputs = x2.unbind(1)
        Da_input = (Da.squeeze(-1) - 0.2) / (0.5 - 0.2)

        if torch.any(x1_inputs[0]<0) or torch.any(x2_inputs[0]<0):
            assert False, "All first inputs must be available"
        
        x1_out = []
        x2_out = []
        
        x1_out.append(x1_inputs[0])
        x2_out.append(x2_inputs[0])
        
        for j in range(len(x1_inputs)):
#             print('is it there?')
#             print(torch.sum((x1_inputs[j]>0)*1))
#             print(torch.sum((x1_inputs[j]<0)*1))

#             print('===========')
#             print(x1_inputs[j].item())
#             print(x2_inputs[j].item())
#             print(x1_out[j].item())
#             print(x2_out[j].item())
            
            if j < warmup:
                x1_input = torch.where(x1_inputs[j]>0,x1_inputs[j],x1_out[j])
                x2_input = torch.where(x2_inputs[j]>0,x2_inputs[j],x2_out[j])
#                 x1_input = x1_inputs[j]
#                 x2_input = x2_inputs[j]
        
            else:
                x1_input = x1_out[j]
                x2_input = x2_out[j]
            
#             print('--------------')
#             print(x1_input.item())
#             print(x2_input.item())
            
#             if j > 5:
#                 assert False
            
            x1_input_norm = (x1_input - self.x1min) / (self.x1max - self.x1min)
            x2_input_norm = (x2_input - self.x2min) / (self.x2max - self.x2min)
            
#             print('shapes')
#             print(Da.squeeze().shape)
#             print(x1_input.shape)
            
            ANN_input = torch.stack((x1_input_norm,x2_input_norm,Da_input),dim=-1)
#             ANN_input = torch.stack((x1_input,x2_input/10,Da.squeeze(-1)/10),dim=-1)
            
#             print('shapes')
#             print(ANN_input.shape)
            
#             ANN_output = self.model(ANN_input)
            
            for layer in self.layers:
                ANN_input = layer(ANN_input)
                ANN_input = self.activation(ANN_input)
                
            ANN_output = self.output_layer(ANN_input)
            
            x1_integ = x1_input + ANN_output[:,0] # * dt.squeeze()
            x2_integ = x2_input + ANN_output[:,1] # * dt.squeeze()
            
            x1_integ = torch.clip(x1_integ, min = 0, max = self.x1max*1000)
            x2_integ = torch.clip(x2_integ, min = 0, max = self.x2max*1000)

#             x1_integ = self.sigmoid(ANN_output[:,0])*1.1
#             x2_integ = self.sigmoid(ANN_output[:,1])*10
    
            x1_out.append(x1_integ)
            x2_out.append(x2_integ)
            
        x1_outs = torch.stack(x1_out[1:],dim=1)
        x2_outs = torch.stack(x2_out[1:],dim=1)
        
        return x1_outs, x2_outs

            

class my_Model():
    def __init__(self, dataloader_train, dataloader_val, network, learning_rate=0.05, device=None):
        if torch.cuda.is_available() and device is None:
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True
            self.device = 'cuda'
        elif not torch.cuda.is_available() and device is None:
            self.device = 'cpu'
        else:
            self.device = device

        print('Using:', self.device)
        
        self.y_loss_mult = config["DATA"]["Y_SAMPLE"]

        self.net = network.to(self.device)

        self.trainable_parameters = \
            sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        print('Trainable parameters: '+str(self.trainable_parameters))

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        
        self.lr = learning_rate
        
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, amsgrad=True)#, weight_decay=0.1)

        self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)

        self.train_loss = []
        self.val_loss = []
        
        self.lr_epoch = []
        self.lr_track = []

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=50, factor=0.5, min_lr=0.000001)

#         self.scheduler_second = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, patience=50, factor=0.5, min_lr=0.000001)
#         self.scheduler_first = CosineAnnealingWarmupRestarts(self.optimizer,
#                                           first_cycle_steps=20,
#                                           cycle_mult=2,
#                                           max_lr=learning_rate,
#                                           min_lr=0.0001,
#                                           warmup_steps=0,
#                                           gamma=1.0)
        
        self.warmup = 0.8
        
        self.alpha = 0.8
        
        self.autoreg_prop = 0
    
#         self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
#                                                   first_cycle_steps=20,
#                                                   cycle_mult=2,
#                                                   max_lr=learning_rate,
#                                                   min_lr=0.001,
#                                                   warmup_steps=0,
#                                                   gamma=1.0)


        self.total_steps = int(np.ceil(config["DATA"]["N_TEST"]/config["TRAINING"]["BATCH_SIZE"]) * (config["TRAINING"]["EPOCHS"]-200))

#         self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=self.total_steps,
#                                                             final_div_factor=1e2,
#                                                             )
#         self.scheduler2 = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=learning_rate/10, total_steps=total_steps,
#                                                             final_div_factor=1e2,
#                                                             )
    
    def train(self, epoch):
        """Train model."""
        self.net.train()
        cnt, sum_loss = 0, 0
        iters = len(self.dataloader_train)
        for (x1, x1out, x2, x2out, dt, Da) in self.dataloader_train:
            self.optimizer.zero_grad()
            
            if epoch == 200:
                self.optimizer = torch.optim.AdamW(
                            self.net.parameters(), lr=self.lr, amsgrad=True)
#                 self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                             self.optimizer, patience=50, factor=0.5, min_lr=0.000001)
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=5e-3, total_steps=self.total_steps,
                                                            final_div_factor=1e2,
                                                            )
            
            batch_size = x1.size()[0]
            time_length = x1.size()[1]
            
            indices = np.random.permutation(batch_size)
            autoreg_indices = indices[:int(batch_size*self.autoreg_prop)]
            teacher_forcing_indices = indices[int(batch_size*self.autoreg_prop):]

            
            if epoch > 200:
#                 warmup = int((np.maximum(0,1-(epoch-50)/(1259-50))*(1-self.warmup)+self.warmup) * x1.size()[1])
                warmup = int(self.warmup * x1.size()[1])
            else:
                warmup = time_length
            
            # Teacher Forcing
            x1out_hat, x2out_hat = self.net(x1[teacher_forcing_indices,:].to(self.device), 
                                            x2[teacher_forcing_indices,:].to(self.device), 
                                            dt[teacher_forcing_indices,:].to(self.device), 
                                            Da[teacher_forcing_indices,:].to(self.device), time_length)
#             loss = self.criterion(x2out.to(self.device)[x2out>0], x2out_hat[x2out>0])
#             loss += self.criterion(x1out.to(self.device)[x1out>0], x1out_hat[x1out>0])# * self.y_loss_mult

#             loss_x2 = self.criterion(
#                     x2out.to(self.device),
#                     torch.where(x2out.to(self.device)>0, x2out_hat, x2out.to(self.device))
#                     ) / 2
#             loss_x1 = self.criterion(
#                     x1out.to(self.device),
#                     torch.where(x1out.to(self.device)>0, x1out_hat, x1out.to(self.device))
#                     ) * self.y_loss_mult / 2

            loss_tf = self.criterion(
                    torch.where(x2out[teacher_forcing_indices,:].to(self.device)>0, x2out[teacher_forcing_indices,:].to(self.device), 0.),
                    torch.where(x2out[teacher_forcing_indices,:].to(self.device)>0, x2out_hat, 0.)
                    ) / 2
            loss_tf += self.criterion(
                    torch.where(x1out[teacher_forcing_indices,:].to(self.device)>0, x1out[teacher_forcing_indices,:].to(self.device), 0.),
                    torch.where(x1out[teacher_forcing_indices,:].to(self.device)>0, x1out_hat, 0.)
                    ) * self.y_loss_mult / 2
#             print('my losses')
#             print(torch.where(x1out.to(self.device)>0, x1out.to(self.device), 0.)[0,:20])
#             print(torch.where(x1out.to(self.device)>0, x1out_hat, 0.)[0,:20])
#             print()
#             assert(False)

            ## Auto regressive ##
#             x1out_hat, x2out_hat = self.net(x1[autoreg_indices,:].to(self.device), 
#                                             x2[autoreg_indices,:].to(self.device), 
#                                             dt[autoreg_indices,:].to(self.device), 
#                                             Da[autoreg_indices,:].to(self.device), warmup)
        
#             loss += self.criterion(
#                     torch.where(x2out[autoreg_indices,:].to(self.device)>0, x2out[autoreg_indices,:].to(self.device), 0.),
#                     torch.where(x2out[autoreg_indices,:].to(self.device)>0, x2out_hat, 0.)
#                     ) / 20
#             loss += self.criterion(
#                     torch.where(x1out[autoreg_indices,:].to(self.device)>0, x1out[autoreg_indices,:].to(self.device), 0.),
#                     torch.where(x1out[autoreg_indices,:].to(self.device)>0, x1out_hat, 0.)
#                     ) * self.y_loss_mult / 20

            x1out_hat, x2out_hat = self.net(x1[teacher_forcing_indices,:].to(self.device), 
                                            x2[teacher_forcing_indices,:].to(self.device), 
                                            dt[teacher_forcing_indices,:].to(self.device), 
                                            Da[teacher_forcing_indices,:].to(self.device), warmup)
    
            loss_autoreg = self.criterion(
                    torch.where(x2out[teacher_forcing_indices,:].to(self.device)>0, x2out[teacher_forcing_indices,:].to(self.device), 0.),
                    torch.where(x2out[teacher_forcing_indices,:].to(self.device)>0, x2out_hat, 0.)
                    ) / 2
            loss_autoreg += self.criterion(
                    torch.where(x1out[teacher_forcing_indices,:].to(self.device)>0, x1out[teacher_forcing_indices,:].to(self.device), 0.),
                    torch.where(x1out[teacher_forcing_indices,:].to(self.device)>0, x1out_hat, 0.)
                    ) * self.y_loss_mult / 2
            
            loss = torch.log(loss_tf) * self.alpha + torch.log(loss_autoreg) * (1 - self.alpha)
            
#             loss = self.criterion(torch.cat((x1out,x2out),dim=-1).to(self.device), torch.cat((x1out_hat,x2out_hat),dim=-1))
#             loss = self.criterion(x2out.to(self.device), x2out_hat)
#             loss += self.criterion(x1out.to(self.device), x1out_hat) * self.y_loss_mult
#             loss = loss/batch_size
            
#             loss = loss / 2
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            self.optimizer.step()
            

            
            # Use for 1-Cycle
            if epoch >= 200:
                self.scheduler.step()
                self.lr_epoch.append(epoch + cnt / iters)
                self.lr_track.append(self.scheduler.get_last_lr())
            
    #        Use for Cosine Annealing
#             self.scheduler.step(epoch + cnt / iters)
#             self.lr_epoch.append(epoch + cnt / iters)
#             self.lr_track.append(self.scheduler.get_lr())
            
#             if epoch <= 2560:
#                 self.scheduler_first.step(epoch + cnt / iters)
#                 self.lr_epoch.append(epoch + cnt / iters)
#                 self.lr_track.append(self.scheduler_first.get_lr())
            
            sum_loss += torch.exp(loss).detach().cpu().numpy()
            cnt += 1
        
#         if epoch > 2560:
#             self.scheduler_second.step(sum_loss / cnt)
#             self.lr_epoch.append(epoch)
#             self.lr_track.append(self.scheduler_second._last_lr)
        
#         Use for lr reduce on plateau
        if epoch < 200:# or epoch <= 1:
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
            for (x1, x1out, x2, x2out, dt, Da) in self.dataloader_train:
                self.optimizer.zero_grad()
                
                batch_size = x1.size()[0]
                time_length = x1.size()[1]

                if epoch > 200:
#                     warmup = int((np.maximum(0,1-(epoch-50)/(1259-50))*(1-self.warmup)+self.warmup) * x1.size()[1])
                    warmup = int(self.warmup * x1.size()[1])
                else:
                    warmup = time_length

                x1out_hat, x2out_hat = self.net(x1.to(self.device), x2.to(self.device), dt.to(self.device), Da.to(self.device), warmup)
#                 loss = self.criterion(x2out.to(self.device)[x2out>0], x2out_hat[x2out>0])
#                 loss += self.criterion(x1out.to(self.device)[x1out>0], x1out_hat[x1out>0])# * self.y_loss_mult

                loss_x2 = self.criterion(
                        torch.where(x2out.to(self.device)>0, x2out.to(self.device), 0.),
                        torch.where(x2out.to(self.device)>0, x2out_hat, 0.)
                        ) / 2
                loss_x1 = self.criterion(
                        torch.where(x1out.to(self.device)>0, x1out.to(self.device), 0.),
                        torch.where(x1out.to(self.device)>0, x1out_hat, 0.)
                        ) * self.y_loss_mult / 2
        
                loss = torch.log(loss_x1 + loss_x2)

#                 loss = self.criterion(torch.cat((x1out,x2out),dim=-1).to(self.device), torch.cat((x1out_hat,x2out_hat),dim=-1))
#                 loss += self.criterion(x1out.to(self.device), x1out_hat) * self.y_loss_mult
#                 loss = loss/batch_size
                
                sum_loss += torch.exp(loss).detach().cpu().numpy()
                cnt += 1


            self.val_loss.append(sum_loss/cnt)
        return sum_loss/cnt


    def save_network(self, name):
        """Save network weights and training loss history."""
        filename = name+'_hidden_layers_' +\
            str(len(self.net.hidden_cells))+'_'+str(self.net.hidden_cells[0])+'.net'
        torch.save(self.net.state_dict(), filename)
        np.save(name+'_training_loss.npy', np.array(self.train_loss))
        np.save(name+'_validation_loss.npy', np.array(self.val_loss))
        np.save(name+'_lr_epoch.npy', np.array(self.lr_epoch))
        np.save(name+'_lr_track.npy', np.array(self.lr_track))
        return name

    def load_network(self, name):
        """Load network weights and training loss history."""
        filename = name+'_hidden_layers_' +\
            str(len(self.net.hidden_cells))+'_'+str(self.net.hidden_cells[0])+'.net'
        self.net.load_state_dict(torch.load(filename))
        self.train_loss = np.load(name+'_training_loss.npy').tolist()
        self.val_loss = np.load(name+'_validation_loss.npy').tolist()


def progress(train_loss, val_loss):
    """Define progress bar description."""
    return "Train/Loss: {:.2e}  Val/Loss: {:.2e}".format(
        train_loss, val_loss)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr