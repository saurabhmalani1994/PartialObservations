import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

from BandFModel.datagen import ode_fun, par_fun, torch_ode_fun
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

from config import config
from main.utils import Network, MLP
import torch
import itertools
from torch.autograd.functional import jacobian

from sklearn.preprocessing import MinMaxScaler

def get_metrics(filename=None):

    ## Get all the data ##
    jactrue, jacpred = metric_Jacobian(filename=filename)
    output, ANN_output, output_LC, ANN_output_LC = metric_RHS(filename=filename)
    RHS_hair_error_trueLC_True, RHS_hair_error_trueLC_Pred = metric_hair_fromTrue(filename=filename)
    RHS_hair_error_predLC_True, RHS_hair_error_predLC_Pred = metric_hair_fromPred(filename=filename)

    # scale all the data #
    jac_scaler = MinMaxScaler
    # jac_scaler.fit(jactrue.reshape())




def metric_Jacobian(filename=None, make_plots=False):
    network = load_model(filename=filename)

    f = np.load('minmax/limitcycle_20.npz')
    myvar0 = f['arr_0']
    f = np.load('minmax/limitcycle_detail.npz')
    myvar0_detail = f['arr_0']

    x = np.linspace(0.99, 1.01, 5)
    p = itertools.product(x, repeat=6)

    p_arr = np.array(list(p)).T

    parr = np.tile(p_arr,(myvar0.shape[1]))
    myvar0arr = np.repeat(myvar0,(p_arr.shape[1]), axis=1)
    RHS_Eval_vals = (parr * myvar0arr).T

    pars = par_fun()
    def torchfun(x):
        return torch_ode_fun(0., x.T, pars).squeeze()

    def batch_jacobian(f, x):
        f_sum = lambda x: torch.sum(f(x), axis=0)
        return jacobian(f_sum, x).permute(1,0,2)

    def torchfun_pred(x):
        return network.output(x.reshape((-1,6)).to(network.device),
                                torch.tensor([6]).reshape((-1,1)).to(network.device)).cpu()

    jactrue = batch_jacobian(torchfun, torch.tensor(RHS_Eval_vals)).numpy()
    jacpred = batch_jacobian(torchfun_pred, torch.tensor(RHS_Eval_vals)).numpy()

    return jactrue, jacpred

def metric_RHS(filename=None, make_plots=False):
    network = load_model(filename=filename)

    f = np.load('minmax/limitcycle_20.npz')
    myvar0 = f['arr_0']

    f = np.load('minmax/limitcycle_detail.npz')
    myvar0_detail = f['arr_0']

    x = np.linspace(0.99, 1.01, 5)
    p = itertools.product(x, repeat=6)
    p_arr = np.array(list(p)).T
    parr = np.tile(p_arr,(myvar0.shape[1]))
    myvar0arr = np.repeat(myvar0,(p_arr.shape[1]), axis=1)
    RHS_Eval_vals = (parr * myvar0arr).T

    RHS_Eval_vals = RHS_Eval_vals.reshape((-1,6))
    ANN_output = network.output(torch.tensor(RHS_Eval_vals).reshape((-1,6)).to(network.device),
                                torch.tensor([6]).reshape((-1,1)).to(network.device)).cpu().detach().numpy()
    pars = par_fun()
    output = ode_fun(0., RHS_Eval_vals.T, pars)

    x = np.linspace(1., 1.01, 1)
    p = itertools.product(x, repeat=6)
    p_arr = np.array(list(p)).T
    parr = np.tile(p_arr,(myvar0_detail.shape[1]))
    myvar0arr = np.repeat(myvar0_detail,(p_arr.shape[1]), axis=1)
    RHS_Eval_vals_LC = (parr * myvar0arr).T

    RHS_Eval_vals_LC = RHS_Eval_vals_LC.reshape((-1,6))
    ANN_output_LC = network.output(torch.tensor(RHS_Eval_vals_LC).reshape((-1,6)).to(network.device),
                                torch.tensor([6]).reshape((-1,1)).to(network.device)).cpu().detach().numpy()

    pars = par_fun()
    output_LC = ode_fun(0., RHS_Eval_vals_LC.T, pars)

    if make_plots:
        fig = plt.figure(figsize=(15,10))

        labels = ['x', 'y', 'z', 'u', 'v', 'g']
        for i in range(6):
            ax = plt.subplot(int(str(23) + str(i+1)))
            axplot = ax.scatter(output[:,i], ANN_output[:,i], s=4)
            axplot = ax.scatter(output_LC[:,i], ANN_output_LC[:,i], s=4)
            ax.plot([np.min(output[:,i]), np.max(output[:,i])],[np.min(output[:,i]), np.max(output[:,i])], 'k-')
            ax.tick_params(axis='y', labelsize=20)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            ax.yaxis.offsetText.set_fontsize(15)
            ax.tick_params(axis='x', labelsize=20) 
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            ax.xaxis.offsetText.set_fontsize(15)
            plt.title(labels[i], fontsize=30)
            ax.set_aspect("equal", adjustable="datalim")

        plt.suptitle('Predicted RHS vs True RHS', fontsize=25)

        fig.supylabel('Predicted RHS', fontsize=25)
        fig.supxlabel('True RHS', fontsize=25)
        plt.tight_layout()
        plt.savefig('Figures/Prediction_of_RHS_parallelepiped.png')
        # plt.show()

    return output, ANN_output, output_LC, ANN_output_LC

def metric_hair_fromTrue(filename=None, make_plots=False):
    network = load_model(filename=filename)

    f = np.load('minmax/limitcycle_50.npz')
    myvar0 = f['arr_0']
    f = np.load('minmax/limitcycle_detail.npz')
    myvar0_detail = f['arr_0']

    def ode_NN_func(t,x,p):
        x_in = torch.tensor(x).reshape((-1,6)).to(network.device)
        p_in = torch.tensor(p).reshape((-1,1)).to(network.device)
        return network.output(x_in, p_in).cpu().detach().numpy()

    if make_plots:
        fig = plt.figure(figsize=(20,10))
        axes = []
        labels = ['x', 'y', 'z', 'u', 'v', 'g']
        for i in range(2):
            axes.append(plt.subplot(int(str(12) + str(i+1))))

    RHS_hair_error_trueLC_True = []
    RHS_hair_error_trueLC_Pred = []

    for j in range(myvar0.shape[1]):
        init = myvar0[:,j]
        # print(init)

        pars = par_fun()

        T=20
        # init[1] = 0
        teval = np.linspace(0,T,1000)

        sol_true = solve_ivp(ode_fun, y0=init, t_span=[0, T],
                                t_eval=teval, args=(pars,),
                                rtol=1e-5, atol=1e-8)

        sol_pred = solve_ivp(ode_NN_func, y0=init, t_span=[0, T],
                                t_eval=teval, args=([6]),
                                rtol=1e-5, atol=1e-8)

        RHS_hair_error_trueLC_True.append(sol_true.y[:,-1])
        RHS_hair_error_trueLC_Pred.append(sol_pred.y[:,-1])

        axes[0].plot(sol_true.y[2,:],sol_true.y[0,:], 'k', label='True ODE', linewidth=2)
        axes[0].plot(sol_pred.y[2,:],sol_pred.y[0,:], 'b', label='Trained Model', linewidth=2)
        axes[0].set_ylabel('x  ', color='k', fontsize=35, rotation=0)
        axes[0].tick_params(axis='y', labelsize=20, direction='in', length=8, width=2)
        axes[0].set_xlabel('z', color='k', fontsize=35)
        axes[0].tick_params(axis='x', labelsize=20, direction='in', length=8, width=2)

        axes[1].plot(sol_true.y[5,:],sol_true.y[0,:], 'k', label='True ODE', linewidth=2)
        axes[1].plot(sol_pred.y[5,:],sol_pred.y[0,:], 'b', label='Trained Model', linewidth=2)
        axes[1].set_ylabel('x  ', color='k', fontsize=35, rotation=0)
        axes[1].tick_params(axis='y', labelsize=20, direction='in', length=8, width=2)
        axes[1].set_xlabel('g', color='k', fontsize=35)
        axes[1].tick_params(axis='x', labelsize=20, direction='in', length=8, width=2)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles[:2], labels[:2], loc='lower center', ncol=2, fontsize=30, bbox_to_anchor= (0.5, -0.05))

    RHS_hair_error_trueLC_True = np.vstack(RHS_hair_error_trueLC_True)
    RHS_hair_error_trueLC_Pred = np.vstack(RHS_hair_error_trueLC_Pred)

    return RHS_hair_error_trueLC_True, RHS_hair_error_trueLC_Pred

def metric_hair_fromPred(filename=None, make_plots=False):
    network = load_model(filename=filename)
    pars = par_fun()

    def ode_NN_func(t,x,p):
        x_in = torch.tensor(x).reshape((-1,6)).to(network.device)
        p_in = torch.tensor(p).reshape((-1,1)).to(network.device)
        return network.output(x_in, p_in).cpu().detach().numpy()

    def fsolve_fun_pred(var):
        T = var[0]
        init = np.array([50])
        init = np.concatenate((init,var[1:]))


        sol = solve_ivp(ode_NN_func, y0=init, t_span=[0, 0.1],
                        args=([6]),
                        rtol=1e-5, atol=1e-8)

        sol = solve_ivp(ode_NN_func, y0=sol.y[:,-1], t_span=[0, T],
                            args=([6]),
                            rtol=1e-5, atol=1e-8)

        return sol.y[:,-1] - init

    fsolve_guess = np.array([150, 0, 0.33, 100, 1000, 0.75])
    root_pred = fsolve(fsolve_fun_pred, fsolve_guess)

    init = np.array([50])
    init = np.concatenate((init, root_pred[1:]))
    T = root_pred[0]
    teval = np.linspace(0,T,100)
    sol_pred = solve_ivp(ode_NN_func, y0=init, t_span=[0, T],
                            t_eval=teval, args=([6]),
                            rtol=1e-5, atol=1e-8)

    if make_plots:
        fig = plt.figure(figsize=(20,10))
        axes = []
        labels = ['x', 'y', 'z', 'u', 'v', 'g']
        for i in range(2):
            axes.append(plt.subplot(int(str(12) + str(i+1))))

    RHS_hair_error_predLC_True = []
    RHS_hair_error_predLC_Pred = []

    for j in range(sol_pred.shape[1]):
        init = sol_pred[:,j]
        # print(init)

        pars = par_fun()

        T=20
        # init[1] = 0
        teval = np.linspace(0,T,1000)

        sol_true = solve_ivp(ode_fun, y0=init, t_span=[0, T],
                                t_eval=teval, args=(pars,),
                                rtol=1e-5, atol=1e-8)

        sol_pred = solve_ivp(ode_NN_func, y0=init, t_span=[0, T],
                                t_eval=teval, args=([6]),
                                rtol=1e-5, atol=1e-8)

        RHS_hair_error_predLC_True.append(sol_true.y[:,-1])
        RHS_hair_error_predLC_Pred.append(sol_pred.y[:,-1])

        axes[0].plot(sol_true.y[2,:],sol_true.y[0,:], 'k', label='True ODE', linewidth=2)
        axes[0].plot(sol_pred.y[2,:],sol_pred.y[0,:], 'b', label='Trained Model', linewidth=2)
        axes[0].set_ylabel('x  ', color='k', fontsize=35, rotation=0)
        axes[0].tick_params(axis='y', labelsize=20, direction='in', length=8, width=2)
        axes[0].set_xlabel('z', color='k', fontsize=35)
        axes[0].tick_params(axis='x', labelsize=20, direction='in', length=8, width=2)

        axes[1].plot(sol_true.y[5,:],sol_true.y[0,:], 'k', label='True ODE', linewidth=2)
        axes[1].plot(sol_pred.y[5,:],sol_pred.y[0,:], 'b', label='Trained Model', linewidth=2)
        axes[1].set_ylabel('x  ', color='k', fontsize=35, rotation=0)
        axes[1].tick_params(axis='y', labelsize=20, direction='in', length=8, width=2)
        axes[1].set_xlabel('g', color='k', fontsize=35)
        axes[1].tick_params(axis='x', labelsize=20, direction='in', length=8, width=2)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles[:2], labels[:2], loc='lower center', ncol=2, fontsize=30, bbox_to_anchor= (0.5, -0.05))

    RHS_hair_error_predLC_True = np.vstack(RHS_hair_error_predLC_True)
    RHS_hair_error_predLC_Pred = np.vstack(RHS_hair_error_predLC_Pred)

    return RHS_hair_error_predLC_True, RHS_hair_error_predLC_Pred

def make_poincare_maps(filename=None, make_plots=False):
    network = load_model(filename=filename)
    pars = par_fun()

    def fsolve_fun_true(var):
        T = var[0]
        init = np.array([50])
        init = np.concatenate((init,var[1:]))


        sol = solve_ivp(ode_fun, y0=init, t_span=[0, 0.1],
                        args=(pars,),
                        rtol=1e-5, atol=1e-8)

        sol = solve_ivp(ode_fun, y0=sol.y[:,-1], t_span=[0, T],
                            args=(pars,),
                            rtol=1e-5, atol=1e-8)

        return sol.y[:,-1] - init

    fsolve_guess = np.array([150, 0, 0.33, 100, 1000, 0.75])
    root_true = fsolve(fsolve_fun_true, fsolve_guess)

    def ode_NN_func(t,x,p):
        x_in = torch.tensor(x).reshape((-1,6)).to(network.device)
        p_in = torch.tensor(p).reshape((-1,1)).to(network.device)
        return network.output(x_in, p_in).cpu().detach().numpy()

    def fsolve_fun_pred(var):
        T = var[0]
        init = np.array([50])
        init = np.concatenate((init,var[1:]))


        sol = solve_ivp(ode_NN_func, y0=init, t_span=[0, 0.1],
                        args=([6]),
                        rtol=1e-5, atol=1e-8)

        sol = solve_ivp(ode_NN_func, y0=sol.y[:,-1], t_span=[0, T],
                            args=([6]),
                            rtol=1e-5, atol=1e-8)

        return sol.y[:,-1] - init

    fsolve_guess = np.array([150, 0, 0.33, 100, 1000, 0.75])
    # fsolve_guess = root_true
    root_pred = fsolve(fsolve_fun_pred, fsolve_guess)

    if make_plots:
        init = np.array([50])
        init = np.concatenate((init, root_true[1:]))
        T = root_true[0]
        # init[1] = 0
        teval = np.linspace(0,T,1000)

        sol_true = solve_ivp(ode_fun, y0=init, t_span=[0, T],
                                t_eval=teval, args=(pars,),
                                rtol=1e-5, atol=1e-8)

        init = np.array([50])
        init = np.concatenate((init, root_pred[1:]))
        T = root_pred[0]
        # init[1] = 0
        teval = np.linspace(0,T,1000)

        sol_pred = solve_ivp(ode_NN_func, y0=init, t_span=[0, T],
                                t_eval=teval, args=([6]),
                                rtol=1e-5, atol=1e-8)


        fig = plt.figure(figsize=(20,10))
        labels = ['x', 'y', 'z', 'u', 'v', 'g']
        for i in range(6):
            ax = plt.subplot(int(str(61) + str(i+1)))
            ax.plot(sol_true.t, sol_true.y[i,:], 'k', label='True ODE')
            ax.plot(sol_pred.t, sol_pred.y[i,:], 'b', label='Trained Model')
            ax.set_ylabel(labels[i], color='k', fontsize=25)
            ax.tick_params(axis='y', labelsize=20)                              
            if i<5:
                ax.set_xticks([])
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=15)

        plt.xlabel('Time', fontsize=25)
        ax.tick_params(axis='x', labelsize=20)

        # plt.figure(figsize=(10,8))
        # ax = plt.subplot(211)
        # ax.plot(sol.y[0,:], sol.y[2,:])

        fig = plt.figure(figsize=(20,7))

        arrow_freq = 20
        true_freq = int(np.floor(sol_true.y[0,:].size/arrow_freq))
        pred_freq = int(np.floor(sol_pred.y[0,:].size/arrow_freq))

        # ax = plt.subplot(131)
        ax = plt.subplot2grid((1, 8), (0, 0), colspan=3)
        ax.plot(sol_true.y[2,:], sol_true.y[0,:], 'k', linewidth=2, label='True')
        ax.plot(sol_pred.y[2,:], sol_pred.y[0,:], 'b', linewidth=2, label='Predicted')
        for i in range(arrow_freq):
            ax.annotate("", xy=(sol_true.y[2,int((i+0.5)*true_freq)+1], sol_true.y[0,int((i+0.5)*true_freq)+1]), 
                            xytext=(sol_true.y[2,int((i+0.5)*true_freq)], sol_true.y[0,int((i+0.5)*true_freq)]),
                    arrowprops=dict(headwidth=10, color='k'))
            ax.annotate("", xy=(sol_pred.y[2,int((i+0.5)*pred_freq)+1], sol_pred.y[0,int((i+0.5)*pred_freq)+1]), 
                            xytext=(sol_pred.y[2,int((i+0.5)*pred_freq)], sol_pred.y[0,int((i+0.5)*pred_freq)]),
                    arrowprops=dict(headwidth=10, color='b'))
        # ax.legend(fontsize=20)
        ax.set_ylabel('x  ', color='k', fontsize=35, rotation=0)
        ax.tick_params(axis='y', labelsize=20, direction='in', length=8, width=2)
        ax.set_xlabel('z', color='k', fontsize=35)
        ax.tick_params(axis='x', labelsize=20, direction='in', length=8, width=2)
        [x.set_linewidth(1.5) for x in ax.spines.values()]

        # ax = plt.subplot(132)
        ax = plt.subplot2grid((1, 8), (0, 3), colspan=3)
        ax.plot(sol_true.y[5,:], sol_true.y[0,:], 'k', linewidth=2, label='True')
        ax.plot(sol_pred.y[5,:], sol_pred.y[0,:], 'b', linewidth=2, label='Predicted')
        for i in range(arrow_freq):
            ax.annotate("", xy=(sol_true.y[5,int((i+0.5)*true_freq)+1], sol_true.y[0,int((i+0.5)*true_freq)+1]), 
                            xytext=(sol_true.y[5,int((i+0.5)*true_freq)], sol_true.y[0,int((i+0.5)*true_freq)]),
                    arrowprops=dict(headwidth=10, color='k'))
            ax.annotate("", xy=(sol_pred.y[5,int((i+0.5)*pred_freq)+1], sol_pred.y[0,int((i+0.5)*pred_freq)+1]), 
                            xytext=(sol_pred.y[5,int((i+0.5)*pred_freq)], sol_pred.y[0,int((i+0.5)*pred_freq)]),
                    arrowprops=dict(headwidth=10, color='b'))
        # ax.legend(fontsize=20)
        ax.set_ylabel('x  ', color='k', fontsize=35, rotation=0)
        ax.tick_params(axis='y', labelsize=20, direction='in', length=8, width=2)
        ax.set_xlabel('g', color='k', fontsize=35)
        ax.tick_params(axis='x', labelsize=20, direction='in', length=8, width=2)
        handles, labels = ax.get_legend_handles_labels()
        [x.set_linewidth(1.5) for x in ax.spines.values()]

        width = 0.35
        # ax = plt.subplot(133)
        ax = plt.subplot2grid((1, 8), (0, 6), colspan=2)
        ax.bar(-width/1.8,root_true[0],width=width,color='k',label='Ground Truth')
        ax.bar(width/1.8,root_pred[0],
                width=width,label='Model Prediction', color='b')
        ax.set_xticks([0])
        # ax.set_yscale('log')
        ax.set_ylim(bottom=0.0)
        ax.set_xticklabels([r"Limit Cycle Period"],fontsize=25)
        ax.tick_params(axis='y', labelsize=20, direction='in', length=8, width=2)
        ax.set_ylabel('Time', color='k', fontsize=25)
        plt.yticks(fontsize=20)
        [x.set_linewidth(1.5) for x in ax.spines.values()]

        # plt.legend(fontsize=20, mode="expand")

        handles, labels = ax.get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=20, bbox_to_anchor= (0.5, -0.1))
        leg.get_frame().set_linewidth(0.0)

        plt.suptitle('Limit Cycle Prediction', fontsize=30)
        plt.tight_layout()
        plt.savefig('Figures/Prediction_LimitCycle.png')

def load_model(filename=None):


    f = np.load("/home/smalani/PartialObservations/minmax/minmax.npz")

    xmax = f['arr_0']
    xmin = f['arr_1']
    
    print('maxmins')
    print(xmax)
    print(xmin)
    
    
    norm_func = lambda input, device: (input - torch.tensor(xmin).float().to(device)) / \
                            ((torch.tensor((xmax - xmin)).float().to(device)) + 1e-10)
    inv_norm_func = lambda input, device: input * ((torch.tensor((xmax - xmin)).float().to(device)) + 1e-10) \
                                  + torch.tensor(xmin).float().to(device)
    
    
    if config["MODEL"]["BOX"] == 'Black':
        # Create the network architecture
        mlp = MLP(6, config["MODEL"]["NUM_HIDDEN"], 6)
        
        class my_Network(Network):
            def __init__(self, network, train_size, xdim, norm_func=lambda input, device: input,
                         inv_norm_func=lambda input, device: input, init_available=True, device=None, 
                         tf_prop=1., integrator='RK4', add_par_num=0):
                super(my_Network, self).__init__(network, train_size, xdim, norm_func,
                         inv_norm_func, init_available, device, 
                         tf_prop, integrator, add_par_num)

                self.additional_pars = torch.nn.Parameter((torch.zeros(6)-1).to(self.device), requires_grad = True) 

            def output(self, x, par):

                # ANN_input = torch.cat((self.norm_func(x), par/20), dim=-1)
                ANN_input_out = self.norm_func(x)
                ANN_input = torch.clip(ANN_input_out, min=-1., max=2.)
                out = self.net(ANN_input)
                out = self.inv_norm_func(out) * (100 ** (self.additional_pars))
                return out
    
    elif config["MODEL"]["BOX"] == 'Grey' or config["MODEL"]["BOX"] == 'Gray':
        
        # Create the network architecture
        mlp = MLP(6, config["MODEL"]["NUM_HIDDEN"], 3)
        
        if config["MODEL"]["Parameters"] == 'Trainable':
            class my_Network(Network):
                def __init__(self, network, train_size, xdim, norm_func=lambda input, device: input,
                             inv_norm_func=lambda input, device: input, init_available=True, device=None, 
                             tf_prop=1., integrator='RK4', add_par_num=2):
                    super(my_Network, self).__init__(network, train_size, xdim, norm_func,
                             inv_norm_func, init_available, device, 
                             tf_prop, integrator, add_par_num)

                    self.additional_pars = torch.nn.Parameter((torch.cat(((torch.tensor([0.4476, -0.4859, -0.6419])), 
                                                                          (torch.zeros(4) + 1)))).to(self.device), 
                                                                requires_grad = True)

                def output(self, x_input, par):
                    ANN_input_out = self.norm_func(x_input)
                    ANN_input = torch.clip(ANN_input_out, min=-1., max=2.)
                    ANN_output = self.net(ANN_input)
                    ANN_output = ANN_output * (100 ** (self.additional_pars[:3]))

                    x, y, z, u, v, g = torch.unbind(x_input, dim=-1)

                    u1_prime_x, u2_prime_y, u3_prime_z = torch.unbind(ANN_output, dim=-1)                

                    omega = self.additional_pars[-4] * 10
                    sigma = self.additional_pars[-3] * 10
                    rho = self.additional_pars[-2] / 10
                    eta = self.additional_pars[-1]

                    alpha, uf, _, _, _, _, _, _, uc1_prime, uc2_prime, uc3_prime = datagen.par_fun()

                    output = []

                    output.append(-alpha * x + u1_prime_x - uc1_prime * x)
                    output.append(-alpha * y + u2_prime_y - uc2_prime * y)
                    output.append(-alpha * z + u3_prime_z - uc3_prime * z)
                    output.append(alpha * (uf - u) - u1_prime_x)
                    output.append(-alpha * v + omega * u1_prime_x - u2_prime_y - sigma * u3_prime_z)
                    output.append(-alpha * g + rho * u2_prime_y + eta * u3_prime_z)

                    out = torch.stack((output), dim=-1)
                    return out
                def raw_output(self, x_input, par):
                    ANN_input_out = self.norm_func(x_input)
                    ANN_input = torch.clip(ANN_input_out, min=-1., max=2.)
                    ANN_output = self.net(ANN_input)
                    ANN_output = ANN_output * (100 ** (self.additional_pars[:3]))
                    return ANN_output
        elif config["MODEL"]["Parameters"] == 'Fixed':
            class my_Network(Network):
                def __init__(self, network, train_size, xdim, norm_func=lambda input, device: input,
                             inv_norm_func=lambda input, device: input, init_available=True, device=None, 
                             tf_prop=1., integrator='RK4', add_par_num=2):
                    super(my_Network, self).__init__(network, train_size, xdim, norm_func,
                             inv_norm_func, init_available, device, 
                             tf_prop, integrator, add_par_num)

                    self.additional_pars = torch.nn.Parameter((torch.tensor([0.4877, -0.7982, -0.6599])).to(self.device), 
                                                                    requires_grad = True)

                def output(self, x_input, par):
                    ANN_input_out = self.norm_func(x_input)
                    ANN_input = torch.clip(ANN_input_out, min=-1., max=2.)
                    ANN_output_out = self.net(ANN_input)
                    ANN_output = ANN_output_out * (100 ** (self.additional_pars))

                    x, y, z, u, v, g = torch.unbind(x_input, dim=-1)
                    u1_prime_x, u2_prime_y, u3_prime_z = torch.unbind(ANN_output, dim=-1)

                    alpha, uf, omega, sigma, rho, eta, phi1, phi2, uc1_prime, uc2_prime, uc3_prime = datagen.par_fun(D=1/7.3, sf=2.5)

                    output = []

                    output.append(-alpha * x + u1_prime_x - uc1_prime * x)
                    output.append(-alpha * y + u2_prime_y - uc2_prime * y)
                    output.append(-alpha * z + u3_prime_z - uc3_prime * z)
                    output.append(alpha * (uf - u) - u1_prime_x)
                    output.append(-alpha * v + omega * u1_prime_x - u2_prime_y - sigma * u3_prime_z)
                    output.append(-alpha * g + rho * u2_prime_y + eta * u3_prime_z)

                    out = torch.stack((output), dim=-1)
                    return out

                def raw_output(self, x_input, par):
                    ANN_input_out = self.norm_func(x_input)
                    ANN_input = torch.clip(ANN_input_out, min=-1., max=2.)
                    ANN_output_out = self.net(ANN_input)
                    ANN_output = ANN_output_out * (100 ** (self.additional_pars))
                    return ANN_output

        else:
            raise ValueError("Tell me whether to train the parameters!")
    else:
        raise ValueError("Tell me what box to use!")

    network = my_Network(mlp, config["DATA"]["N_TRAIN"], 6, norm_func=norm_func, inv_norm_func=inv_norm_func, 
                      init_available=config["DATA"]["INIT_AVAILABLE"], integrator='RK4')
    device = 'cpu'

    if filename is None:
        filename = 'data/' + 'model_' + '.net'

    print(filename)
    state_dict = torch.load(filename, map_location=torch.device(device))
    network.load_state_dict(state_dict, strict=False)

    print(network)
    network.double()

    return network