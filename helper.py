import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


import numbers
import math
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.double)

from torch import Tensor
import torch.jit as jit

from datagen import f_cstr, get_pars
from utils import Network, MLP
from config import config
import datagen
import preprocess

# from utils import f_cstr, get_pars, integrate_cstr, Network, config, CSTRDataset

def myf_cstr_torch(t, y, Da, beta, B):
    """Time derivatives of the CSTR model."""
    x1, x2 = y
    dx1dt = -x1 + Da * (1-x1) * torch.exp(x2)
    dx2dt = -x2 + B * Da * (1-x1) * torch.exp(x2) - beta * x2

    return torch.stack((dx1dt, dx2dt))

def make_Bifurc():
    unstable_ss_true, unstable_Da_true, stable_ss1_true, stable_Da1_true, stable_ss2_true, stable_Da2_true,\
    Da_arr_true, x1min_true, x2min_true, x1max_true, x2max_true = make_Bifurc_true()
    
    unstable_ss_pred, unstable_Da_pred, stable_ss_pred, stable_Da_pred, \
    Da_arr_pred, x1min_pred, x2min_pred, x1max_pred, x2max_pred = make_Bifurc_prediction()
    
    
#     print('My bifurc shapes')
#     print(stable_ss_pred.shape)
#     print(stable_Da_pred.shape)
#     print(stable_ss_pred)
#     print(stable_Da_pred)
#     print('===================')
    
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111)
    ax1.plot(stable_Da1_true,stable_ss1_true[:,0],'k',label='True')
    ax1.plot(stable_Da2_true,stable_ss2_true[:,0],'k')
    ax1.plot(unstable_Da_true,unstable_ss_true[:,0],'k--')
    ax1.plot(Da_arr_true, x1min_true, 'k')
    ax1.plot(Da_arr_true, x1max_true, 'k')

    if stable_Da_pred.size > 0:
        ax1.plot(stable_Da_pred,stable_ss_pred[...,0],'b',label='Prediction')
    if unstable_Da_pred.size > 0:
        ax1.plot(unstable_Da_pred,unstable_ss_pred[...,0],'b--')
    ax1.plot(Da_arr_pred, x1min_pred, 'b')
    ax1.plot(Da_arr_pred, x1max_pred, 'b')

    plt.legend(fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_xlabel(r'Da', fontsize=24)
    ax1.set_ylabel(r'$x_1$', fontsize=24)
    fig.savefig('Figures/Bifurcation Plot x1.png',format='png')

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111)
    ax1.plot(stable_Da1_true,stable_ss1_true[:,1],'k',label='True')
    ax1.plot(stable_Da2_true,stable_ss2_true[:,1],'k')
    ax1.plot(unstable_Da_true,unstable_ss_true[:,1],'k--')
    ax1.plot(Da_arr_true, x2min_true, 'k')
    ax1.plot(Da_arr_true, x2max_true, 'k')

    if stable_Da_pred.size > 0:
        ax1.plot(stable_Da_pred,stable_ss_pred[...,1],'b',label='Prediction')
    if unstable_Da_pred.size > 0:
        ax1.plot(unstable_Da_pred,unstable_ss_pred[...,1],'b--')
    ax1.plot(Da_arr_pred, x2min_pred, 'b')
    ax1.plot(Da_arr_pred, x2max_pred, 'b')

    plt.legend(fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_xlabel(r'Da', fontsize=24)
    ax1.set_ylabel(r'$x_2$', fontsize=24)
    fig.savefig('Figures/Bifurcation Plot x2.png',format='png')

def make_Bifurc_true(verbose=False):
    Da_arr = np.linspace(0.2,0.5,100)
    
    stable_ss1 = []
    stable_Da1 = []
    
    stable_ss2 = []
    stable_Da2 = []
    
    unstable_ss = []
    unstable_Da = []
    
    x1max = []
    x1min = []
    
    x2max = []
    x2min = []
    
    roots = []
    switch = False
    root = [0.5, 3]
    for i in range(len(Da_arr)):
        Da = Da_arr[i]
        
        pars = get_pars(Da)
        
        numpy_f_cstr = lambda y: f_cstr(0, y, *pars)
        numpy_f_cstr_integ = lambda t, y: f_cstr(0, y, *pars)
        torch_f_cstr = lambda y: myf_cstr_torch(0, y, *pars)
        
        root = fsolve(numpy_f_cstr, root)
        J = torch.autograd.functional.jacobian(torch_f_cstr, torch.from_numpy(root), strict=True)
        w, v = np.linalg.eig(J)
        
        
        if np.any(w.real>0):
            switch = True
            unstable_ss.append(root)
            unstable_Da.append(Da)
            
            solved = False
            perturb = 0.1
            
            y0 = [root[1], 2]
            while not solved:
                x10 = root[0] + perturb
#                 y0 = [root[1], 2]

                y, infodict, ier, mesg = fsolve(ODE_Bifurc, y0, args=(numpy_f_cstr_integ, Da, x10), full_output=True)
                
                if ier == 1:
                    solved = True
                    y0 = y
                else:
                    perturb = perturb * 0.5
                    
            y0_int = [x10, y[0]]
            t_eval = np.linspace(0, y[1], 1000)

            sol = solve_ivp(f_cstr, y0=y0_int, t_span=[0, t_eval[-1]],
                                args=pars, t_eval=t_eval,
                                rtol=1e-5, atol=1e-8)

            x1max.append(np.max(sol.y[0,:]))
            x2max.append(np.max(sol.y[1,:]))
            x1min.append(np.min(sol.y[0,:]))
            x2min.append(np.min(sol.y[1,:]))
                    
        else:
            if not switch:
                stable_ss1.append(root)
                stable_Da1.append(Da)
                
                x1max.append(root[0])
                x2max.append(root[1])
                x1min.append(root[0])
                x2min.append(root[1])
            else:
                stable_ss2.append(root)
                stable_Da2.append(Da)
                
                x1max.append(root[0])
                x2max.append(root[1])
                x1min.append(root[0])
                x2min.append(root[1])
        
        roots.append(root)
    roots = np.array(roots)
    unstable_ss = np.array(unstable_ss)
    unstable_Da = np.array(unstable_Da)
    stable_ss1 = np.array(stable_ss1)
    stable_Da1 = np.array(stable_Da1)
    stable_ss2 = np.array(stable_ss2)
    stable_Da2 = np.array(stable_Da2)
    
    if verbose:
    
        fig = plt.figure(figsize=(15,15))
        ax1 = fig.add_subplot(111)
        ax1.plot(stable_Da1,stable_ss1[:,0],'k')
        ax1.plot(stable_Da2,stable_ss2[:,0],'k')
        ax1.plot(unstable_Da,unstable_ss[:,0],'k--')
        ax1.plot(Da_arr, x1min, 'k')
        ax1.plot(Da_arr, x1max, 'k')
        ax1.set_xlabel(r'Da', fontsize=30)
        ax1.set_ylabel(r'$x_1$', fontsize=30)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        fig.savefig('data/Bifurcation Plot True x1.png',format='png')

        fig = plt.figure(figsize=(15,15))
        ax1 = fig.add_subplot(111)
        ax1.plot(stable_Da1,stable_ss1[:,1],'k')
        ax1.plot(stable_Da2,stable_ss2[:,1],'k')
        ax1.plot(unstable_Da,unstable_ss[:,1],'k--')
        ax1.plot(Da_arr, x2min, 'k')
        ax1.plot(Da_arr, x2max, 'k')
        ax1.set_xlabel(r'Da', fontsize=30)
        ax1.set_ylabel(r'$x_2$', fontsize=30)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        fig.savefig('data/Bifurcation Plot True x2.png',format='png')
    return unstable_ss, unstable_Da, stable_ss1, stable_Da1, stable_ss2, stable_Da2, Da_arr, x1min, x2min, x1max, x2max
    
def ODE_Bifurc(y, func, Da, x10):
    x2, T = y
    pars = get_pars(Da)
    
#     event = ODE_Event
#     event.terminal = True
    
    y0 = [x10, x2]

    
    sol = solve_ivp(func, y0=y0, t_span=[0, 0.1],
#                     args=pars,
                    rtol=1e-5, atol=1e-8, dense_output=True)#, events=(event,))#, dense_output=True)
    
    y_init = sol.y[:,-1]
    
    sol = solve_ivp(func, y0=y_init, t_span=[0.1, T],
#                     args=pars,
                    rtol=1e-5, atol=1e-8, dense_output=True)#, events=(event,))#, dense_output=True)
    
    T_out = sol.t[-1]
    x1_out = sol.y[0,-1]
    x2_out = sol.y[1,-1]

    return (x10-x1_out), (x2-x2_out)

def make_Bifurc_prediction(verbose=False):
    network = load_network()
#     network(self, x1_input: Tensor, x2_input: Tensor, Da: Tensor)
    
    
    
    Da_arr = np.linspace(0.2,0.5,100)
    
    stable_ss = []
    stable_Da = []
    
    unstable_ss = []
    unstable_Da = []
    
    x1max = []
    x1min = []
    
    x2max = []
    x2min = []
    
    roots = []
    switch = False
    root = [0.5, 1]
    for i in range(len(Da_arr)):
        Da = Da_arr[i]
        
        pars = get_pars(Da)
        
        torch_function = lambda f: network.output(f.unsqueeze(0).to(network.device),
                                                   torch.tensor([Da]).unsqueeze(0).to(network.device))
        numpy_function = lambda f: network.output(torch.tensor(f).unsqueeze(0).to(network.device),
                                                   torch.tensor([Da]).unsqueeze(0).to(network.device)).detach().cpu().squeeze().numpy()
        numpy_function_integ = lambda t, f: network.output(torch.tensor(f).unsqueeze(0).to(network.device),
                                                   torch.tensor([Da]).unsqueeze(0).to(network.device)).detach().cpu().squeeze().numpy()
        
        root = fsolve(numpy_function, root)
        J = torch.autograd.functional.jacobian(torch_function, torch.from_numpy(root), strict=True)
        w, v = np.linalg.eig(J)

        if np.any(w.real>0):
            if switch is not True:
                switch = True
                stable_ss.append(np.array([np.nan, np.nan]))
                stable_Da.append(np.nan)
                
            unstable_ss.append(root)
            unstable_Da.append(Da)

            solved = False
            perturb = 0.1
            
            y0 = [root[1], 2]
            while not solved:
                x10 = root[0] + perturb
                

                y, infodict, ier, mesg = fsolve(ODE_Bifurc, y0, args=(numpy_function_integ, Da, x10), full_output=True)
                
                if ier == 1:
                    solved = True
                    y0=y
                else:
                    perturb = perturb * 0.8
                    
            y0_int = [x10, y[0]]
            t_eval = np.linspace(0, y[1], 1000)

            sol = solve_ivp(numpy_function_integ, y0=y0_int, t_span=[0, t_eval[-1]],
                                t_eval=t_eval,
                                rtol=1e-5, atol=1e-8)
            
            x1max.append(np.max(sol.y[0,:]))
            x2max.append(np.max(sol.y[1,:]))
            x1min.append(np.min(sol.y[0,:]))
            x2min.append(np.min(sol.y[1,:]))
                    
        else:
            if switch is not False:
                switch = False
                unstable_ss.append(np.array([np.nan, np.nan]))
                unstable_Da.append(np.nan)
            stable_ss.append(root)
            stable_Da.append(Da)

            x1max.append(root[0])
            x2max.append(root[1])
            x1min.append(root[0])
            x2min.append(root[1])

        
        roots.append(root)
    roots = np.array(roots)
    
    
    unstable_ss = np.array(unstable_ss)
    unstable_Da = np.array(unstable_Da)
    stable_ss = np.array(stable_ss)
    stable_Da = np.array(stable_Da)
    
    if verbose:

        plt.figure()
    #     plt.plot(Da_arr,roots[:,0])
        if stable_Da.size > 0:
            plt.plot(stable_Da,stable_ss[...,0],'b')
        if unstable_Da.size > 0:
            plt.plot(unstable_Da,unstable_ss[...,0],'b--')
        plt.plot(Da_arr, x1min, 'b')
        plt.plot(Da_arr, x1max, 'b')
        plt.xlabel(r'Da', fontsize=24)
        plt.ylabel(r'$x_1$', fontsize=24)

        plt.figure()
    #     plt.plot(Da_arr,roots[:,1])
        if stable_Da.size > 0:
            plt.plot(stable_Da,stable_ss[...,1],'b')
        if unstable_Da.size > 0:
            plt.plot(unstable_Da,unstable_ss[...,1],'b--')
        plt.plot(Da_arr, x2min, 'b')
        plt.plot(Da_arr, x2max, 'b')
        plt.xlabel(r'Da', fontsize=24)
        plt.ylabel(r'$x_2$', fontsize=24)
        
    return unstable_ss, unstable_Da, stable_ss, stable_Da, Da_arr, x1min, x2min, x1max, x2max
    
    
    
def load_network():
    
#     xmaxmin = np.savez("minmax/minmax.npz",xmax, xmin)
    
    f = np.load("minmax/minmax.npz")
    
    xmax = f['arr_0']
    xmin = f['arr_1']
    print(xmax)
    print(xmin)
    
    norm_func = lambda input, device: (input - torch.tensor(xmin).float().to(device)) / \
                            torch.tensor((xmax - xmin)).float().to(device)
    inv_norm_func = lambda input, device: input * torch.tensor((xmax - xmin)).float().to(device) \
                                  + torch.tensor(xmin).float().to(device)

    # Create the network architecture
    if config["MODEL"]["BOX"] == 'Black':
        # Create the network architecture
        mlp = MLP(3, config["MODEL"]["NUM_HIDDEN"], 2)
        
        class my_Network(Network):
            def __init__(self, network, train_size, xdim, norm_func=lambda input, device: input,
                         inv_norm_func=lambda input, device: input, init_available=True, device=None, 
                         tf_prop=1., integrator='RK4', add_par_num=0):
                super(my_Network, self).__init__(network, train_size, xdim, norm_func,
                         inv_norm_func, init_available, device, 
                         tf_prop, integrator, add_par_num)

            def output(self, x, par):

                ANN_input = torch.cat((self.norm_func(x), par), dim=-1)
                out = self.net(ANN_input)
                out = torch.stack((out[...,0] / 10,
                                   out[...,1],
                                 ), dim=-1)
                return out
    
    elif config["MODEL"]["BOX"] == 'Grey' or config["MODEL"]["BOX"] == 'Gray':
        
        # Create the network architecture
        mlp = MLP(3, config["MODEL"]["NUM_HIDDEN"], 1)
        
        if config["MODEL"]["Parameters"] == 'Trainable':
            class my_Network(Network):
                def __init__(self, network, train_size, xdim, norm_func=lambda input, device: input,
                             inv_norm_func=lambda input, device: input, init_available=True, device=None, 
                             tf_prop=1., integrator='RK4', add_par_num=2):
                    super(my_Network, self).__init__(network, train_size, xdim, norm_func,
                             inv_norm_func, init_available, device, 
                             tf_prop, integrator, add_par_num)

                def output(self, x, par):

                    ANN_input = torch.cat((self.norm_func(x), par), dim=-1)
                    g = self.net(ANN_input)[...,0]

                    x1 = x[...,0]
                    x2 = x[...,1]
                    B = self.additional_pars[0] * 10
                    beta = self.additional_pars[1] * 10

                    dx1dt = -x1 + g
                    dx2dt = -x2 + B * g - beta * x2

                    out = torch.stack((dx1dt,dx2dt), dim=-1)
                    return out
        elif config["MODEL"]["Parameters"] == 'Fixed':
            class my_Network(Network):
                def __init__(self, network, train_size, xdim, norm_func=lambda input, device: input,
                             inv_norm_func=lambda input, device: input, init_available=True, device=None, 
                             tf_prop=1., integrator='RK4', add_par_num=0):
                    super(my_Network, self).__init__(network, train_size, xdim, norm_func,
                             inv_norm_func, init_available, device, 
                             tf_prop, integrator, add_par_num)
                    self.fixed_parameters = torch.tensor([11, 3]).to(self.device)
                def output(self, x, par):

                    ANN_input = torch.cat((self.norm_func(x), par), dim=-1)
                    g = self.net(ANN_input)[...,0]

                    x1 = x[...,0]
                    x2 = x[...,1]
                    B = self.fixed_parameters[0]
                    beta = self.fixed_parameters[1]

                    dx1dt = -x1 + g
                    dx2dt = -x2 + B * g - beta * x2

                    out = torch.stack((dx1dt,dx2dt), dim=-1)
                    return out
        else:
            raise ValueError("Tell me whether to train the parameters!")
    else:
        raise ValueError("Tell me what box to use!")
    network = my_Network(mlp, config["DATA"]["N_TRAIN"], 2, norm_func=norm_func, inv_norm_func=inv_norm_func, 
                      init_available=config["DATA"]["INIT_AVAILABLE"], integrator='RK4')
    device = 'cpu'

    filename = config["DATA"]["PATH"]+'model_' + '.net'

    print(filename)

#         omit_dict = {"initial_x1": "initial_x1", "initial_x2": "initial_x2"}
#         model_dict = network.state_dict()
#         model_dict.pop("initial_x1")
#         model_dict.pop("initial_x2")

#         chosen_dict = {k: v for k, v in omit_dict.items() if k in model_dict}
#         chosen_dict = {k: v for k, v in model_dict.items() if k not in omit_dict}

#         model_dict.update(chosen_dict)
    state_dict = torch.load(filename, map_location=torch.device(device))
#     state_dict.pop('initial_x')
#         network.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    network.load_state_dict(state_dict, strict=False)
#     network.initial_x1 = torch.tensor(dataset_test.x1[0][0]/2).unsqueeze(0).unsqueeze(0)
#     network.initial_x2 = torch.tensor(dataset_test.x2[0][0]/10).unsqueeze(0).unsqueeze(0)

    print(network)
#     network.to(device)

    network.double()
    return network


def make_RHS(Da_list = [0.2, 0.25, 0.28, 0.3, 0.33, 0.36, 0.4, 0.42, 0.45, 0.5]):
    
    real_RHS_x1 = []
    real_RHS_x2 = []
    real_RHS_x1_pred = []
    real_RHS_x2_pred = []
    
    pred_RHS_x1 = []
    pred_RHS_x2 = []
    pred_RHS_x1_pred = []
    pred_RHS_x2_pred = []
    
    real_phi = []
    real_phi_pred = []
    pred_phi = []
    pred_phi_pred = []
    
    network = load_network()
    
    for i in range(len(Da_list)):
        Da = Da_list[i]
        
        data_test = datagen.generate_data(n_train=config["DATA"]["N_TEST"],
                                              tmax=config["DATA"]["TMAX"]*6,                
                                              x1_sample_num=config["DATA"]["X1_SAMPLE_NUM"]*6,
                                              x2_sample_num=config["DATA"]["X2_SAMPLE_NUM"]*6, 
                                        init_available=True,
                                        Da_random=False,
                                        Da_set=Da,
                                         detail=True)
#         dataset_test = preprocess.Dataset(data_test[3],data_test[1])

        if config["MODEL"]["BOX"]== 'Grey' or config["MODEL"]["BOX"] == 'Gray':
            def phi_network(x_in_RHS, p_in_RHS):
                return network.raw_output(x_in_RHS, p_in_RHS)

        x1_in_RHS = torch.from_numpy(data_test[3][...,0])
        x2_in_RHS = torch.from_numpy(data_test[3][...,1])
        x_in_RHS = torch.from_numpy(data_test[3]).to(network.device)
        p_in_RHS = torch.from_numpy(data_test[4]).unsqueeze(-1).to(network.device)

        myB, mybeta, myD = 11, 3, torch.from_numpy(data_test[4])#
        real_RHS_x1.append((-x1_in_RHS + myD * (1-x1_in_RHS) * torch.exp(x2_in_RHS)).detach().cpu().squeeze().numpy())
        real_RHS_x2.append((-x2_in_RHS + myB * myD * (1-x1_in_RHS) * torch.exp(x2_in_RHS) - mybeta * x2_in_RHS).detach().cpu().squeeze().numpy())
        
        ANN_output = network.output(x_in_RHS, p_in_RHS)
        pred_RHS_x1.append(ANN_output[...,0].detach().cpu().squeeze().numpy())
        pred_RHS_x2.append(ANN_output[...,1].detach().cpu().squeeze().numpy())
        
        
        if config["MODEL"]["BOX"] == 'Grey' or config["MODEL"]["BOX"] == 'Gray':
            real_phi.append((myD * (1-x1_in_RHS) * torch.exp(x2_in_RHS)).detach().cpu().squeeze().numpy())
            ANN_output = phi_network(x_in_RHS, p_in_RHS)
            pred_phi.append(ANN_output[...,0].detach().cpu().squeeze().numpy())
            
            
    real_RHS_x1 = np.concatenate((np.array(real_RHS_x1,dtype=object))).flatten()
    real_RHS_x2 = np.concatenate((np.array(real_RHS_x2,dtype=object))).flatten()
    pred_RHS_x1 = np.concatenate((np.array(pred_RHS_x1,dtype=object))).flatten()
    pred_RHS_x2 = np.concatenate((np.array(pred_RHS_x2,dtype=object))).flatten()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    sc = ax.scatter(real_RHS_x1, pred_RHS_x1)
    ax.plot([np.min(real_RHS_x1),np.max(real_RHS_x1)],[np.min(real_RHS_x1),np.max(real_RHS_x1)],'k-')
    ax.set_xlabel('True RHS: ' + r'$x_1$',fontsize=40)
    ax.set_ylabel('Predicted RHS: ' + r'$x_1$',fontsize=40)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    fig.savefig('Figures/RHS for x1' + '.png',format='png', bbox_inches='tight')
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    sc = ax.scatter(real_RHS_x2, pred_RHS_x2)
    ax.plot([np.min(real_RHS_x2),np.max(real_RHS_x2)],[np.min(real_RHS_x2),np.max(real_RHS_x2)],'k-')
    ax.set_xlabel('True RHS: ' + r'$x_2$',fontsize=40)
    ax.set_ylabel('Predicted RHS: ' + r'$x_2$',fontsize=40)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    fig.savefig('Figures/RHS for x2' + '.png',format='png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    if config["MODEL"]["BOX"] == 'Grey' or config["MODEL"]["BOX"] == 'Gray':
        real_phi = np.concatenate((np.array(real_phi,dtype=object))).flatten()
        pred_phi = np.concatenate((np.array(pred_phi,dtype=object))).flatten()
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.plot(real_phi, pred_phi,'b.')
        ax.plot([np.min(real_phi),np.max(real_phi)],[np.min(real_phi),np.max(real_phi)],'k-')
        ax.set_xlabel('True phi: ',fontsize=40)
        ax.set_ylabel('Predicted phi: ',fontsize=40)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        fig.savefig('Figures/Phi predictions' + '.png',format='png', bbox_inches='tight')
        
        if config["MODEL"]["Parameters"] == 'Trainable':
            labels = [r'$B$',r'$\beta$']
            x = np.arange(len(labels))
            width = 0.35
            
            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111)
            ax.bar(x-width/2,[11, 3],width=width,label='Ground Truth')
            ax.bar(x+width/2,[network.additional_pars[0] * 10, 
                              network.additional_pars[1] * 10],
                   width=width,label='Model Prediction')
            ax.set_xticks(x)
            ax.set_xticklabels(labels,fontsize=30)
            plt.yticks(fontsize=30)
            plt.title('Model Prediction of Experimental Parameters',fontsize=25)
            plt.legend(fontsize=30)
            fig.savefig('Figures/Prediction of Experiment Parameters' + '.png',format='png')
    
    
def make_transients(Da_list = [0.2, 0.25, 0.28, 0.3, 0.33, 0.36, 0.4, 0.42, 0.45, 0.5]):
    network = load_network()
    index = 0
    
    for i in range(len(Da_list)):
        Da = Da_list[i]

        data_test = datagen.generate_data(n_train=config["DATA"]["N_TEST"],
                                              tmax=config["DATA"]["TMAX"]*6,                
                                              x1_sample_num=config["DATA"]["X1_SAMPLE_NUM"]*6,
                                              x2_sample_num=config["DATA"]["X2_SAMPLE_NUM"]*6, 
                                        init_available=True,
                                        Da_random=False,
                                        Da_set=Da,
                                         detail=True)
        dataset_test = preprocess.Dataset(*data_test[:2])
        output_detail_t, output_detail, output_detail_Da = data_test[2:]
        
#         print('shapes')
#         print(output_detail_t.shape)
#         print(output_detail.shape)

        network.initial_x = torch.nn.Parameter(network.norm_func(torch.tensor(dataset_test.x).float().to(network.device)))
        
        def my_ode(t, y, p):
#             print()
#             y_in = torch.cat((network.norm_func(torch.tensor(y).to(network.device)),
#                                torch.tensor(p).to(network.device)), dim=-1)
            
            dxdt = network.output(torch.tensor(y).to(network.device),
                                  torch.tensor(p).to(network.device))
            
            return dxdt.detach().cpu().squeeze().numpy()
        
#         print('Dataset Shape')
#         print(dataset_test.x.shape)
#         print(dataset_test.full_times_arr[0].shape)
        
        y_in = dataset_test.x[0,0,:]
        solver_time = np.linspace(0,dataset_test.full_times_arr[0][-1],1000)
        sol = solve_ivp(my_ode,[0,dataset_test.full_times_arr[0][-1]], y_in, args=(dataset_test.p[0,0,:],), t_eval = solver_time,
                    rtol=1e-5, atol=1e-8)
        
        x1_out = sol.y[0,:]
        x2_out = sol.y[1,:]
        
        fig = plt.figure(figsize=(20,10))
#         fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(111)
        ax.plot(dataset_test.full_times_arr[0][~np.isnan(dataset_test.x[0,:,0])],dataset_test.x[0,:,0][~np.isnan(dataset_test.x[0,:,0])],'x',label='true (training points)',markersize=20,markeredgecolor='#1f77b4',markeredgewidth=2)
        ax.plot(output_detail_t[0,:], output_detail[0,:,0],'k',label='true trajectory',linewidth=3)
        ax.plot(solver_time,x1_out,'x-',label='predicted',linewidth=3)
        ax.set_xlabel(r'$t$',fontsize=40)
        ax.set_ylabel(r'$X_1$',fontsize=40)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        plt.legend(fontsize=20,loc='upper left')
        plt.title(r'$X_1$' + ' graph for Da: ' + str(dataset_test.p[0,0,0]),fontsize=30)
        fig.savefig('Figures/Prediction for x1 for Da_' + str(dataset_test.p[0,0,0]) + 'index_' + str(index) + '_solveivp' + '.png',format='png')
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.plot(dataset_test.full_times_arr[0][~np.isnan(dataset_test.x[0,:,1])],dataset_test.x[0,:,1][~np.isnan(dataset_test.x[0,:,1])],'x',label='true (training points)',markersize=20,markeredgecolor='#1f77b4',markeredgewidth=2)
        ax.plot(output_detail_t[0,:], output_detail[0,:,1],'k',label='true trajectory',linewidth=3)
        ax.plot(solver_time,x2_out,'x-',label='predicted',linewidth=3)
        ax.set_xlabel(r'$t$',fontsize=40)
        ax.set_ylabel(r'$X_2$',fontsize=40)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        plt.legend(fontsize=20,loc='upper left')
        plt.title(r'$X_2$' + ' graph for Da: ' + str(dataset_test.p[0,0,0]),fontsize=30)
        fig.savefig('Figures/Prediction for x2 for Da_' + str(dataset_test.p[0,0,0]) + 'index_' + str(index) + '_solveivp' + '.png',format='png')
        plt.show()
        plt.close()
        
#         

#         x1_in = torch.from_numpy(dataset_test.x1[0]).unsqueeze(0)
#         x2_in = torch.from_numpy(dataset_test.x2[0]).unsqueeze(0)
        
#         x1_out, x2_out = network(x1_in,x2_in,warmup=0,time=torch.tensor(dataset_test.time[0]).unsqueeze(0),Da=torch.tensor(dataset_test.Da).unsqueeze(0),index=None)

#         fig = plt.figure(figsize=(20,10))
#         ax = fig.add_subplot(111)
#         ax.plot(dataset_test.time[0][:][dataset_test.x1_out[0]>0],dataset_test.x1_out[0][dataset_test.x1_out[0]>0],'x',label='true (training points)',markersize=20,markeredgecolor='#1f77b4',markeredgewidth=2)
#         ax.plot(dataset_test.time_detail,dataset_test.x1_data_detail[0],'k',label='true trajectory',linewidth=3)
#         ax.plot(dataset_test.time[0][:],x1_out.detach().cpu().squeeze().numpy(),'x-',label='predicted',linewidth=3)
#         ax.set_xlabel(r'$t$',fontsize=40)
#         ax.set_ylabel(r'$X_1$',fontsize=40)
#         plt.yticks(fontsize=25)
#         plt.xticks(fontsize=25)
#         plt.legend(fontsize=20,loc='upper left')
#         plt.title(r'$X_1$' + ' graph for Da: ' + str(dataset_test.Da),fontsize=30)
#         fig.savefig('Figures/Prediction for x1 for Da_' + str(dataset_test.Da) + 'index_' + str(index) + '_network' + '.png',format='png')
#         plt.show()
#         plt.close()


#         fig = plt.figure(figsize=(20,10))
#         ax = fig.add_subplot(111)
#         ax.plot(dataset_test.time[0][:][dataset_test.x2_out[0]>0],dataset_test.x2_out[0][dataset_test.x2_out[0]>0],'x',label='true (training points)',markersize=20,markeredgecolor='#1f77b4',markeredgewidth=2)
#         ax.plot(dataset_test.time_detail,dataset_test.x2_data_detail[0],'k',label='true trajectory',linewidth=3)
#         ax.plot(dataset_test.time[0][:],x2_out.detach().cpu().squeeze().numpy(),'x-',label='predicted',linewidth=3)
#         ax.set_xlabel(r'$t$',fontsize=40)
#         ax.set_ylabel(r'$X_2$',fontsize=40)
#         plt.yticks(fontsize=25)
#         plt.xticks(fontsize=25)
#         plt.legend(fontsize=20,loc='upper left')
#         plt.title(r'$X_2$' + ' graph for Da: ' + str(dataset_test.Da[0]),fontsize=30)
#         fig.savefig('Figures/Prediction for x2 for Da_' + str(dataset_test.Da) + 'index_' + str(index) + '_network' + '.png',format='png')
#         plt.show()
#         plt.close()
        