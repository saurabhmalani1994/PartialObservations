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
config["DATA"]["TMAX"] = 6
config["DATA"]["L_TRAJECTORIES"] = 200 # 200 equal lengths is dt = 0.02
config["DATA"]["N_TRAIN"] = 83
config["DATA"]["N_VAL"] = 10
config["DATA"]["N_TEST"] = 1
config["DATA"]["PATH"] = 'data/'
config["DATA"]["SKIP_TIME"] = 1
config["DATA"]["X1_SAMPLE_NUM"] = 4
config["DATA"]["X2_SAMPLE_NUM"] = 20
config["DATA"]["MAX_DELTA_T"] = 0.1
config["DATA"]["REG_TIME"] = False   # True for regular sampling in time, False for random sampling in time
config["DATA"]["INIT_AVAILABLE"] = False   # True for ALL initial conditions available at time 0, False for Not available

config["PAR"] = {}
config["PAR"]["Da"] = 0.33
config["PAR"]["B"] = 11.
config["PAR"]["beta"] = 3.

config["TRAINING"] = {}
config["TRAINING"]["BATCH_SIZE"] = 256
config["TRAINING"]["LEARNING_RATE"] = 5e-2
config["TRAINING"]["EPOCHS"] = 2000 # 1000 # 1259 # 2539 # 5260

config["MODEL"] = {}
config["MODEL"]["NUM_HIDDEN"] = [64, 64]

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
#     print(y)
    x1, x2 = y
    dx1dt = -x1 + Da * (1-x1) * np.exp(x2)
    dx2dt = -x2 + B * Da * (1-x1) * np.exp(x2) - beta * x2

    return [dx1dt, dx2dt]


def get_pars(Da: np.float=0.25, B: np.float=11, beta: np.float=3):
#     B = 22.0
#     beta = 3.0
    

    pars = Da, beta, B
    return pars


def integrate_cstr(tmin=0, tmax=20, T=2000, y0=None, verbose=False, Da=0.25, B=11, beta=3, teval=np.linspace(0, 20, 2001)):

    pars = get_pars(Da,B,beta)
    if y0 is None:
        y0 = cstr_initial_conditions()

    sol = solve_ivp(f_cstr, y0=y0, t_span=[tmin, tmax],
                    t_eval=teval, args=pars,
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

    def __init__(self, n_train, tmax, l_trajectories, Da=0.25, B=11, beta=3, maxdt=0.2, random=False, x1_sample_num=50, x2_sample_num=50, skip_time=1, reg_time=True, inference=False):
        self.ids = np.arange(n_train)
        self.x1 = []
        self.x1_out = []
        self.x1_data = []
        self.x1_data_detail = []
        self.x2 = []
        self.x2_out = []
        self.x2_data = []
        self.x2_data_detail = []
        self.time = []

        
        if n_train == 1:
            self.Da = np.array([Da])
        else:
            if random:
                self.Da_base = np.random.uniform(0.2,0.5,10)   
#                 self.Da = np.concatenate((np.random.uniform(0.2,0.22,int(0.2*n_train)),np.random.uniform(0.22,0.5,n_train-int(0.2*n_train))))
#                 self.Da = np.random.uniform(0.33,0.33,n_train)   
            else:
                self.Da_base = np.linspace(0.2,0.5,10)
#                 self.Da = np.concatenate((np.linspace(0.2,0.22,int(0.2*n_train)),np.linspace(0.22,0.5,n_train-int(0.2*n_train))))
#                 self.Da = np.linspace(0.33,0.33,n_train)   
            self.Da = np.random.choice(self.Da_base, n_train, replace=True)
        count = 0
        
        x1_sampling_times_arr = []
        x2_sampling_times_arr = []
        full_times_arr = []
        max_arr_length = 0
        
        for i in range(n_train):
            if reg_time:
                x1_sampling_times = np.linspace(skip_time, skip_time+tmax, num=x1_sample_num+1, endpoint=True).round(decimals=3)
                x2_sampling_times = np.linspace(skip_time, skip_time+tmax, num=x2_sample_num+1, endpoint=True).round(decimals=3)
                solver_sampling_times = np.sort(np.unique(np.concatenate((x1_sampling_times, x2_sampling_times))))
                full_times = self.insert_intermediate(solver_sampling_times, maxdt)
                
            else:
                if inference:
                    x1_sampling_times = np.array([skip_time])
                    x2_sampling_times = np.array([skip_time])
                else:
                    x1_sampling_times = np.array([])
                    x2_sampling_times = np.array([])

                while len(x1_sampling_times) < x1_sample_num:
                    x1_sampling_times = np.append(x1_sampling_times, np.array([np.random.uniform(low=skip_time, high=tmax+skip_time+1e-10)]).round(decimals=3))
                    x1_sampling_times = np.sort(np.unique(x1_sampling_times))

                while len(x2_sampling_times) < x2_sample_num:
                    x2_sampling_times = np.append(x2_sampling_times, np.array([np.random.uniform(low=skip_time, high=tmax+skip_time+1e-10)]).round(decimals=3))
                    x2_sampling_times = np.sort(np.unique(x2_sampling_times))
                
#                 x1_sampling_times = x2_sampling_times
                
                solver_sampling_times = np.sort(np.unique(np.concatenate((np.array([skip_time]), x1_sampling_times, x2_sampling_times))))
                full_times = self.insert_intermediate(solver_sampling_times, maxdt)


            x1_sampling_times_arr.append(x1_sampling_times)
            x2_sampling_times_arr.append(x2_sampling_times)
            full_times_arr.append(full_times)

            if full_times.size > max_arr_length:
                max_arr_length = full_times.size
                
        for i in range(n_train):
            while len(full_times_arr[i]) < max_arr_length:
                full_times_arr[i] = np.append(full_times_arr[i], np.array([tmax+skip_time+0.1]).round(decimals=3))
            self.time.append(full_times_arr[i] - full_times_arr[i][0])
        
        for _ in tqdm(range(n_train), leave=True, position=0):
#             index = int(np.floor(count/10))
            index = count
            x1_sampling_times = x1_sampling_times_arr[index]
            x2_sampling_times = x2_sampling_times_arr[index]
            full_times = full_times_arr[index]
            sol = integrate_cstr(tmin=0, tmax=tmax+skip_time+0.1, T=l_trajectories, Da=self.Da[index], B=B, beta=beta, teval=np.unique(full_times))
            sol_detail = integrate_cstr(tmin=skip_time, tmax=tmax+skip_time, T=l_trajectories, Da=self.Da[index], B=B, beta=beta, teval=np.linspace(skip_time,tmax+skip_time,int(tmax*20)), y0 = sol.y[:,0])
            
            sol_t = sol.t.round(decimals=3)
            
            x1 = np.zeros(len(full_times)) - 1
            x2 = np.zeros(len(full_times)) - 1
            
            x1[np.in1d(full_times,x1_sampling_times)] = sol.y[0, np.in1d(sol_t,x1_sampling_times)]
            x2[np.in1d(full_times,x2_sampling_times)] = sol.y[1, np.in1d(sol_t,x2_sampling_times)]
            
            self.x1.append(x1[:-1])  # x1 at time t except last
            self.x1_out.append(x1[:])  # x1 at all times
            
            self.x2.append(x2[:-1])  # x2 at t except last
            self.x2_out.append(x2[:])  # x2 at all times
            
            self.x1_data.append(sol.y[0, :].copy())
            self.x2_data.append(sol.y[1, :].copy())
            
            self.x1_data_detail.append(sol_detail.y[0, :].copy())
            self.x2_data_detail.append(sol_detail.y[1, :].copy())
            
            count += 1
        self.solver_time = solver_sampling_times - solver_sampling_times[0]
        self.time_detail = sol_detail.t - sol_detail.t[0]
        
#         self.delta_t = self.tt[1]-self.tt[0]  # delta t
        
#         print('Using dt of '+str(self.delta_t))
    
    def insert_intermediate(self, t_samp, max_dt):
        t_arr = []
        for i in range(len(t_samp)-1):
            t_arr.append(np.arange(start=t_samp[i],stop=t_samp[i+1],step=max_dt))
        t_arr.append(np.array([t_samp[-1]]))
        return np.sort(np.unique(np.concatenate(t_arr).round(decimals=3).flatten()))
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x1 = self.x1[self.ids[index]]
        x1_out = self.x1_out[self.ids[index]]
        x2 = self.x2[self.ids[index]]
        x2_out = self.x2_out[self.ids[index]]
        time = self.time[self.ids[index]]
        Da = self.Da[self.ids[index]]
        return torch.tensor(x1, dtype=torch.float64), \
            torch.tensor(x1_out, dtype=torch.float64), \
            torch.tensor(x2, dtype=torch.float64), \
            torch.tensor(x2_out, dtype=torch.float64), \
            torch.tensor(time, dtype=torch.float64), \
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
                - alpha: trainable parameter
                alpha is initialized with one value by default
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
        ## x + (1/a) * sin^2(a*x) ##
        return input_tensor + (1/self.alpha) * (torch.sin(input_tensor * self.alpha)) ** 2 
        
# class Network(nn.Module):
class Network(jit.ScriptModule):
    def __init__(self, hidden_cells, minmaxes, beta=3., B=11., Da=0.33, batch=83, device=None):
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
        
        ############################################################
        ##                        Settings                        ##
        ############################################################
        
        ## Euler, RK2 or RK4 ##
        self.integrator = 'RK2'
        
        ## Black or Grey/Gray ##
        self.box = 'Black'
        
        ## If Grey-Box, Are Parameters Trainable or Fixed ##
        self.parameter_knowledge = 'Fixed'
        
        ############################################################
        
        self.layers = nn.ModuleList()
        self.hidden_cells = hidden_cells
        
        self.layers.append(nn.Linear(3,hidden_cells[0]))
        nn.init.kaiming_normal_(self.layers[-1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.layers[-1].bias, 0)
        for i in range(len(hidden_cells)-1):
            self.layers.append(nn.Linear(hidden_cells[i],hidden_cells[i+1]))
            nn.init.kaiming_normal_(self.layers[-1].weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(self.layers[-1].bias, 0)
            
        if self.box == 'Black':
            self.output_layer = nn.Linear(hidden_cells[-1],2)
            nn.init.xavier_normal_(self.output_layer.weight, gain=nn.init.calculate_gain('linear'))
            nn.init.constant_(self.output_layer.bias, 0)
            
            self.ANNPower1 = nn.Parameter(torch.tensor(-0.5), requires_grad = True)
            self.ANNPower2 = nn.Parameter(torch.tensor(0.), requires_grad = True)
        elif self.box == 'Grey' or self.box == 'Gray':
            self.output_layer = nn.Linear(hidden_cells[-1],1)
            nn.init.xavier_normal_(self.output_layer.weight, gain=nn.init.calculate_gain('linear'))
            nn.init.constant_(self.output_layer.bias, 0)
            
            self.ANNPower1 = nn.Parameter(torch.tensor(-0.08), requires_grad = True)
            self.ANNPower2 = nn.Parameter(torch.tensor(0.), requires_grad = False)
        else:
            assert False, 'No Box of this color'
            
        self.initial_x1 = nn.Parameter(torch.zeros(batch,1), requires_grad = False)
        self.initial_x2 = nn.Parameter(torch.zeros(batch,1), requires_grad = False)
        self.is_initialized = False

        self.x1min, self.x1max, self.x2min, self.x2max = minmaxes
    
        #         self.activation = Snake()
        #         self.activation = nn.Mish()
        self.activation = nn.SiLU()
        #         self.activation = nn.Tanh()
        #         self.activation = nn.SELU()
        
        self.sigmoid = nn.Sigmoid()
        
        if self.parameter_knowledge == 'Fixed' or self.box == 'Black':
            self.B = nn.Parameter(torch.tensor(B/100), requires_grad = False)
            self.beta = nn.Parameter(torch.tensor(beta/100), requires_grad = False)
        elif self.parameter_knowledge == 'Trainable':
            self.B = nn.Parameter(torch.tensor(7./100), requires_grad = True)
            self.beta = nn.Parameter(torch.tensor(7./100), requires_grad = True)
        else:
            assert False ,'Tell me whether or not to train the parameters'
        
        
    @jit.script_method
    def network(self, x1_input: Tensor, x2_input: Tensor, Da: Tensor) -> Tensor:        
        x1_input_norm = (x1_input - self.x1min) / (self.x1max - self.x1min)
        x2_input_norm = (x2_input - self.x2min) / (self.x2max - self.x2min)
        Da_input = (Da.repeat(1,x1_input.size()[1]) - 0.2) / (0.5 - 0.2)
        
        ANN_input = torch.stack((x1_input_norm,x2_input_norm,Da_input),dim=-1)

        for layer in self.layers:
            ANN_input = layer(ANN_input)
            ANN_input = self.activation(ANN_input)
        ANN_output = self.output_layer(ANN_input)
        
        
        if self.box == 'Black':
            return self.BlackBox(ANN_output)
        elif self.box == 'Grey' or self.box == 'Gray':
            return self.GreyBox(x1_input, x2_input, ANN_output)
        else:
            assert False, 'No Box of this Color'
    
    @jit.script_method
    def BlackBox(self, ANN_output: Tensor) -> Tensor:
        #         dx1dt = ANN_output[...,0] * (2.0 + 0.5) / self.hidden_cells[-1] - 0.5
        #         dx2dt = ANN_output[...,1] * (20. + 5.) / self.hidden_cells[-1] - 5.
        
        dx1dt = ANN_output[...,0] * torch.pow(2.,self.ANNPower1*7)
        dx2dt = ANN_output[...,1] * torch.pow(2.,self.ANNPower2*7)
        
        output = torch.stack((dx1dt, dx2dt), dim=-1)
        
        return output
    
    @jit.script_method
    def GreyBox(self, x1: Tensor, x2: Tensor, ANN_output: Tensor) -> Tensor:
        ## TO IMPLEMENT ##
        phi = ANN_output[...,0] * torch.pow(2.,self.ANNPower1*7)  # phi = Da * (1-x1) * np.exp(x2)
        #         phi = ANN_output[...,0] * 10  # phi = Da * (1-x1) * np.exp(x2)

        #         dx1dt = -x1 + Da * (1-x1) * np.exp(x2)
        #         dx2dt = -x2 + B * Da * (1-x1) * np.exp(x2) - beta * x2
        
        dx1dt = - x1 + phi
        dx2dt = - x2 + (self.B*100) * phi - (self.beta*100) * x2
        #         dx2dt = - x2 + 11 * phi - 3 * x2
        
        output = torch.stack((dx1dt, dx2dt), dim=-1)
        
        return output
    
    @jit.script_method
    def Euler(self, x1: Tensor, x2: Tensor, dt: Tensor, Da: Tensor) -> Tuple[Tensor, Tensor]:
        
        ANN_output = self.network(x1, x2, Da)
        
        x1_out = x1 + ANN_output[...,0] * dt
        x2_out = x2 + ANN_output[...,1] * dt 
        
        return x1_out, x2_out
    
    @jit.script_method
    def RK2(self, x1: Tensor, x2: Tensor, dt: Tensor, Da: Tensor) -> Tuple[Tensor, Tensor]:
        x1_input1 = x1
        x2_input1 = x2
        k1 = self.network(x1_input1, x2_input1, Da)
        
        x1_input2 = x1 + k1[...,0] * dt
        x2_input2 = x2 + k1[...,1] * dt
        k2 = self.network(x1_input2, x2_input2, Da)
        
        x1_out = x1 + (dt/2) * (k1[...,0] + k2[...,0])
        x2_out = x2 + (dt/2) * (k1[...,1] + k2[...,1])
    
        #         x1_out = torch.clip(x1_out, min = 0, max = self.x1max*1000)
        #         x2_out = torch.clip(x2_out, min = 0, max = self.x2max*1000)
        return x1_out, x2_out
    
    @jit.script_method
    def RK4(self, x1: Tensor, x2: Tensor, dt: Tensor, Da: Tensor) -> Tuple[Tensor, Tensor]:
        x1_input1 = x1
        x2_input1 = x2
        k1 = self.network(x1_input1, x2_input1, Da)
        
        x1_input2 = x1 + k1[...,0] * dt / 2
        x2_input2 = x2 + k1[...,1] * dt / 2
        k2 = self.network(x1_input2, x2_input2, Da)
        
        x1_input3 = x1 + k2[...,0] * dt / 2
        x2_input3 = x2 + k2[...,1] * dt / 2
        k3 = self.network(x1_input3, x2_input3, Da)
        
        x1_input4 = x1 + k3[...,0] * dt
        x2_input4 = x2 + k3[...,1] * dt
        k4 = self.network(x1_input4, x2_input4, Da)
        
        x1_out = x1 + (dt/6) * (k1[...,0] + 2*k2[...,0] + 2*k3[...,0] + k4[...,0])
        x2_out = x2 + (dt/6) * (k1[...,1] + 2*k2[...,1] + 2*k3[...,1] + k4[...,1])
    
        #         x1_out = torch.clip(x1_out, min = 0, max = self.x1max*1000)
        #         x2_out = torch.clip(x2_out, min = 0, max = self.x2max*1000)

        return x1_out, x2_out
    
    @jit.script_method
    def forward_full(self, x1: Tensor, x2: Tensor, dt: Tensor, Da: Tensor, warmup: float) -> Tuple[Tensor, Tensor]:
        """Forward pass"""
        #         Da_input = Da.repeat(1,x1.size()[1])
        Da_input = Da
    
        if self.integrator == 'Euler':
            x1_out, x2_out = self.Euler(x1, x2, dt, Da_input)
        elif self.integrator == 'RK4':
            x1_out, x2_out = self.RK4(x1, x2, dt, Da_input)
        elif self.integrator == 'RK2':
            x1_out, x2_out = self.RK2(x1, x2, dt, Da_input)
        else:
            assert False, 'Specify Valid Integrator'
            
        x1_out = torch.cat((self.initial_x1, x1_out), dim=1)
        x2_out = torch.cat((self.initial_x2, x2_out), dim=1)
        
        return x1_out, x2_out
    
    @jit.script_method
    def forward_manual(self, x1: Tensor, x2: Tensor, dt: Tensor, Da: Tensor, warmup: float) -> Tuple[Tensor, Tensor]:
        """Forward pass"""
        # x1 = (N, T, 1) N-batches, T-time, dim 1
        
        x1_inputs = x1.unbind(1)
        x2_inputs = x2.unbind(1)
        dt_inputs = dt.unbind(1)
        Da_input = Da#.squeeze(-1)
        
        x1_out = []
        x2_out = []
#         x1_out.append(self.initial_x1.squeeze(1))
#         x2_out.append(self.initial_x2.squeeze(1))
        
        x1_out.append(torch.where(x1_inputs[0]>0,x1_inputs[0],
                                  self.initial_x1.squeeze(1) * (self.x1max - self.x1min) + self.x1min))
        x2_out.append(torch.where(x2_inputs[0]>0,x2_inputs[0],
                                  self.initial_x2.squeeze(1) * (self.x2max - self.x2min) + self.x2min))
        
        switch = int(len(x1_inputs) * warmup) # (1-warmup)
        for j in range(len(x1_inputs)):
#             if torch.remainder(torch.tensor(j),torch.tensor(switch)) < 1:
            if j < switch: # Teacher Forcing
                x1_input = torch.where(x1_inputs[j]>0,x1_inputs[j],x1_out[j])
                x2_input = torch.where(x2_inputs[j]>0,x2_inputs[j],x2_out[j])
            else: # Autoregressive
                x1_input = x1_out[j]
                x2_input = x2_out[j]
            
#             print(x1.shape)
#             print(x2.shape)
            
#             print(x1_out[j].shape)
#             print(x2_out[j].shape)
            
#             print(x1_input.shape)
#             print(x2_input.shape)
#             assert False
            
            dt_input = dt_inputs[j]
            
            if self.integrator == 'Euler':
                x1_integ, x2_integ = self.Euler(x1_input.unsqueeze(-1), x2_input.unsqueeze(-1), dt_input.unsqueeze(-1), Da_input)
            elif self.integrator == 'RK4':
                x1_integ, x2_integ = self.RK4(x1_input.unsqueeze(-1), x2_input.unsqueeze(-1), dt_input.unsqueeze(-1), Da_input)
            elif self.integrator == 'RK2':
                x1_integ, x2_integ = self.RK2(x1_input.unsqueeze(-1), x2_input.unsqueeze(-1), dt_input.unsqueeze(-1), Da_input)
            else:
                assert False, 'Specify Valid Integrator'
    
            x1_out.append(x1_integ.squeeze(-1))
            x2_out.append(x2_integ.squeeze(-1))
            
        x1_outs = torch.stack(x1_out[:],dim=1)
        x2_outs = torch.stack(x2_out[:],dim=1)
        
        return x1_outs, x2_outs
    
    @jit.script_method
    def forward(self, x1: Tensor, x2: Tensor, time: Tensor, Da: Tensor, warmup: float) -> Tuple[Tensor, Tensor]:
#         if self.initial_x1 is None:
#             self.initial_x1 = nn.Parameter(torch.where(x1[:,[0],...] > 0, x1[:,[0],...], (self.x1max-self.x1min)/2), requires_grad = True)
#         if self.initial_x2 is None:
#             self.initial_x2 = nn.Parameter(torch.where(x2[:,[0],...] > 0, x2[:,[0],...], (self.x2max-self.x2min)/2), requires_grad = True)
    
        dt = time[...,1:] - time[...,:-1]
        
        if warmup < 1 or torch.any(x1[0]<0) or torch.any(x2[0]<0):
            return self.forward_manual(x1, x2, dt, Da, warmup)
        else:
            return self.forward_full(x1, x2, dt, Da, warmup)            

class my_Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(my_Loss, self).__init__()
        self.reduction = reduction
        
    def forward(self, input, target):
        count = torch.sum((input>1e-10)*1,dim=1)

        output = (input - target) ** 2
        
        if self.reduction == 'mean':
            return torch.mean(torch.sum(output,dim=1) / count)
        elif self.reduction == 'sum':
            return torch.sum(output)
        else:
            assert False, 'invalid reduction'
        
        
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
        
#         self.x1_loss_mult = config["DATA"]["X1_SAMPLE_TIME"]
#         self.x2_loss_mult = config["DATA"]["X2_SAMPLE_TIME"]
        self.x1_loss_mult = 1
        self.x2_loss_mult = 1

        self.net = network.to(self.device)

        self.trainable_parameters = \
            sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        print('Trainable parameters: '+str(self.trainable_parameters))

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        
        
        f = open('minmax/maxmin.txt', 'r')
        self.x1max = float(f.readline())
        self.x1min = float(f.readline())
        self.x2max = float(f.readline())
        self.x2min = float(f.readline())
        
        self.lr = learning_rate
        
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, amsgrad=True, weight_decay=0.01)
#         self.optimizer = torch.optim.Adam(
#             self.net.parameters(), lr=self.lr)

#         self.criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.criterion = my_Loss(reduction='mean')

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
        
        self.warmup = 0.0
        
        self.alpha = 0.5
        
        self.autoreg_prop = 0
    
#         self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
#                                                   first_cycle_steps=20,
#                                                   cycle_mult=2,
#                                                   max_lr=learning_rate,
#                                                   min_lr=0.001,
#                                                   warmup_steps=0,
#                                                   gamma=1.0)
        self.epoch_shift = 200


        self.total_steps = int(np.ceil(config["DATA"]["N_TEST"]/config["TRAINING"]["BATCH_SIZE"]) * (config["TRAINING"]["EPOCHS"]-self.epoch_shift))

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
        for (x1, x1out, x2, x2out, time, Da) in self.dataloader_train:
            if self.net.is_initialized is False:
                initial_x1 = (torch.zeros(x1[:,[0],...].shape)+0.5).to(self.net.device)
                initial_x2 = (torch.zeros(x2[:,[0],...].shape)+0.5).to(self.net.device)
                
                self.net.initial_x1 = nn.Parameter(initial_x1, requires_grad=True) 
                self.net.initial_x2 = nn.Parameter(initial_x2, requires_grad=True) 
                self.net.is_initialized = True
                
                print('initial x1/x2 Tensors')
                print(self.net.initial_x1)
                print(self.net.initial_x2)
                
                self.optimizer.add_param_group({"params": self.net.initial_x1})
                self.optimizer.add_param_group({"params": self.net.initial_x2})

                self.trainable_parameters = \
                    sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                
                print('Network is initialized with trainable parameter')
                print('Trainable parameters: '+str(self.trainable_parameters))

                my_list = ['initial_x1', 'initial_x2']                
                params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, self.net.named_parameters()))))
                base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, self.net.named_parameters()))))
                
                self.optimizer = torch.optim.AdamW([
                {'params': base_params},
                {'params': params, 'lr': self.lr/10}],
                    lr=self.lr, amsgrad=True, weight_decay=0.01)
#                 self.optimizer = torch.optim.AdamW(
#                     self.net.parameters(), lr=self.lr, amsgrad=True, weight_decay=0.01)
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, patience=50, factor=0.5, min_lr=0.000001)
            
            self.optimizer.zero_grad()
            
            if epoch == self.epoch_shift:
                my_list = ['initial_x1', 'initial_x2']                
                params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, self.net.named_parameters()))))
                base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, self.net.named_parameters()))))
                
                self.optimizer = torch.optim.AdamW([
                {'params': base_params},
                {'params': params, 'lr': self.lr/10}],
                    lr=self.lr, amsgrad=True, weight_decay=0.01)
#                 self.optimizer = torch.optim.Adam(
#                             self.net.parameters(), lr=self.lr)
#                 self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                             self.optimizer, patience=50, factor=0.5, min_lr=0.000001)
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=[1e-2,1e-3],
                                                  total_steps=self.total_steps, final_div_factor=1e2,
                                                            )
                print('Shifting to Autoregressive')
                self.trainable_parameters = \
                    sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                print('Trainable parameters: '+str(self.trainable_parameters))
            
            batch_size = x1.size()[0]
            time_length = x1.size()[1]
            
            indices = np.random.permutation(batch_size)
            autoreg_indices = indices[:int(batch_size*self.autoreg_prop)]
            teacher_forcing_indices = indices[int(batch_size*self.autoreg_prop):]

            
            if epoch > self.epoch_shift:
#                 warmup = int((np.maximum(0,1-(epoch-50)/(1259-50))*(1-self.warmup)+self.warmup) * x1.size()[1])
#                 warmup = int(self.warmup * x1.size()[1])
                warmup = self.warmup
            else:
#                 warmup = time_length
                warmup = 1
            
            # Teacher Forcing
            x1out_hat, x2out_hat = self.net(x1.to(self.device), 
                                            x2.to(self.device), 
                                            time.to(self.device), 
                                            Da.to(self.device), time_length)
            
            x1_norm = (x1out - self.x1min) / (self.x1max - self.x1min)
            x2_norm = ((x2out - self.x2min) / (self.x2max - self.x2min)) 
            x1_hat_norm = (x1out_hat - self.x1min) / (self.x1max - self.x1min)
            x2_hat_norm = ((x2out_hat - self.x2min) / (self.x2max - self.x2min)) 

            loss_tf = self.criterion(
                    torch.where(x2out.to(self.device)>0, x2_norm.to(self.device), 0.),
                    torch.where(x2out.to(self.device)>0, x2_hat_norm, 0.)
                    ) * self.x2_loss_mult / 2
            loss_tf += self.criterion(
                    torch.where(x1out.to(self.device)>0, x1_norm.to(self.device), 0.),
                    torch.where(x1out.to(self.device)>0, x1_hat_norm, 0.)
                    ) * self.x1_loss_mult / 2

            x1out_hat, x2out_hat = self.net(x1.to(self.device), 
                                            x2.to(self.device), 
                                            time.to(self.device), 
                                            Da.to(self.device), warmup)

            x1_norm = (x1out - self.x1min) / (self.x1max - self.x1min)
            x2_norm = ((x2out - self.x2min) / (self.x2max - self.x2min)) 
            x1_hat_norm = (x1out_hat - self.x1min) / (self.x1max - self.x1min)
            x2_hat_norm = ((x2out_hat - self.x2min) / (self.x2max - self.x2min)) 
    
            loss_autoreg = self.criterion(
                    torch.where(x2out.to(self.device)>0, x2_norm.to(self.device), 0.),
                    torch.where(x2out.to(self.device)>0, x2_hat_norm, 0.)
                    ) * self.x2_loss_mult / 2
            loss_autoreg += self.criterion(
                    torch.where(x1out.to(self.device)>0, x1_norm.to(self.device), 0.),
                    torch.where(x1out.to(self.device)>0, x1_hat_norm, 0.)
                    ) * self.x1_loss_mult / 2
            
            loss = torch.exp(torch.log(loss_tf) * self.alpha + torch.log(loss_autoreg) * (1 - self.alpha))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()
            

            
            # Use for 1-Cycle
            if epoch >= self.epoch_shift:
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
            
#             sum_loss += torch.exp(loss).detach().cpu().numpy()
            sum_loss += loss.detach().cpu().numpy()
            cnt += 1
        
#         if epoch > 2560:
#             self.scheduler_second.step(sum_loss / cnt)
#             self.lr_epoch.append(epoch)
#             self.lr_track.append(self.scheduler_second._last_lr)
        
#         Use for lr reduce on plateau
        if epoch < self.epoch_shift:# or epoch <= 1:
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
            for (x1, x1out, x2, x2out, time, Da) in self.dataloader_train:
                self.optimizer.zero_grad()
                
                batch_size = x1.size()[0]
                time_length = x1.size()[1]

                if epoch > 200:
#                     warmup = int((np.maximum(0,1-(epoch-50)/(1259-50))*(1-self.warmup)+self.warmup) * x1.size()[1])
#                     warmup = int(self.warmup * x1.size()[1])
                    warmup = 0
                else:
#                     warmup = time_length
                    warmup = 1.

                x1out_hat, x2out_hat = self.net(x1.to(self.device), x2.to(self.device), time.to(self.device), Da.to(self.device), warmup)


                x1_norm = (x1out - self.x1min) / (self.x1max - self.x1min)
                x2_norm = ((x2out - self.x2min) / (self.x2max - self.x2min)) 
                x1_hat_norm = (x1out_hat - self.x1min) / (self.x1max - self.x1min)
                x2_hat_norm = ((x2out_hat - self.x2min) / (self.x2max - self.x2min)) 

                loss_x2 = self.criterion(
                        torch.where(x2out.to(self.device)>0, x2_norm.to(self.device), 0.),
                        torch.where(x2out.to(self.device)>0, x2_hat_norm, 0.)
                        ) * self.x2_loss_mult / 2
                loss_x1 = self.criterion(
                        torch.where(x1out.to(self.device)>0, x1_norm.to(self.device), 0.),
                        torch.where(x1out.to(self.device)>0, x1_hat_norm, 0.)
                        ) * self.x1_loss_mult / 2
        
                loss = torch.log(loss_x1 + loss_x2)
                
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