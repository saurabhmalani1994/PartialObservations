import sys

from click import style
sys.path.append("..")

from . import helper
from numpy.random import default_rng
import numpy as np
from config import config
from main.utils import Network, MLP
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from .datagen import f_cstr, get_pars
np.random.seed(1234)


from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def metric_ExpPar(filename=None, make_plots=False):
    Da_sample, x1_sample, x2_sample = cloud_sample()
    network = load_network(filename)
    B_pred = network.additional_pars[0].detach().cpu().squeeze().numpy() * 10
    beta_pred = network.additional_pars[1].detach().cpu().squeeze().numpy() * 10
    B_true = 11
    beta_true = 3
    error = 0.5 * np.abs(B_pred - B_true) / B_true + 0.5 * np.abs(beta_pred - beta_true) / beta_true

    return B_true, B_pred, beta_true, beta_pred, error

def metric_Phi(filename=None, make_plots=False):
    Da_sample, x1_sample, x2_sample = cloud_sample()
    network = load_network(filename)

    x1_sample = np.array(x1_sample)
    x2_sample = np.array(x2_sample)
    Da_sample = np.array(Da_sample)

    x_in = torch.from_numpy(np.vstack((x1_sample,x2_sample)).T).to(network.device)
    p_in = torch.from_numpy(Da_sample).unsqueeze(-1).to(network.device)

    myB, mybeta, myD = 11, 3, np.array(Da_sample)
    real_Phi = (Da_sample * (1-x1_sample) * np.exp(x2_sample)).reshape(-1,1)

    ANN_output = network.raw_output(x_in, p_in).detach().cpu().squeeze().numpy()

    pred_Phi = ANN_output.copy().reshape(-1,1)

    scaler = MinMaxScaler()
    scaler.fit(real_Phi)
    
    real_Phi_norm = scaler.transform(real_Phi)
    pred_Phi_norm = scaler.transform(pred_Phi)

    L1_norm = np.sum(np.sum(np.abs(real_Phi_norm - pred_Phi_norm), axis=1)) / real_Phi_norm.size
    L2_norm = np.sum(np.sqrt(np.sum((real_Phi_norm - pred_Phi_norm) ** 2, axis=1))) / real_Phi_norm.size
    Linf_norm = np.sum(np.max(np.abs(real_Phi_norm - pred_Phi_norm), axis=1)) / real_Phi_norm.shape[1]

    if make_plots:
        fig, ax1 = plt.subplots(1,1, figsize=(10,5))
        ax1.scatter(real_Phi_norm[:,0], pred_Phi_norm[:,0], s=1)
        ax1.plot([min(real_Phi_norm[:,0]), max(real_Phi_norm[:,0])], \
            [min(real_Phi_norm[:,0]), max(real_Phi_norm[:,0])], 'k-', lw=2) 
        ax1.set_xlabel('Real Phi x1', fontsize=15)
        ax1.set_ylabel('Predicted Phi x1', fontsize=15)
        ax1.xaxis.set_tick_params(labelsize=12, direction='in')
        ax1.yaxis.set_tick_params(labelsize=12, direction='in')

        fig.title('Phi Prediction', fontsize=20)
        plt.show()


    return real_Phi, pred_Phi, L1_norm, L2_norm, Linf_norm

def metric_hairFromTrue(filename=None, make_plots=False):
    unstable_ss, unstable_Da, stable_ss1, stable_Da1, stable_ss2, stable_Da2, Da_arr, x1min, x2min, x1max, x2max, x1LC, x2LC = \
            make_Bifurc_true_with_LC()
    network = load_network(filename)

    Da_sampler = default_rng(seed=123)
    LC_sampler = default_rng(seed=456)

    Da_sample = np.zeros(100)
    x1_sample = np.zeros(100)
    x2_sample = np.zeros(100)

    def my_ode(t, y, p):        
        dxdt = network.output(torch.tensor(y).to(network.device),
                                torch.tensor(p).to(network.device))
        
        return dxdt.detach().cpu().squeeze().numpy()

    if make_plots:
        x1_plot = []
        x2_plot = []
        Da_plot = []

        for i in range(len(x1LC)):
            x1_plot.append(x1LC[i])
            x2_plot.append(x2LC[i])
            Da_plot.append(np.zeros(np.array(x1LC[i]).shape) +  Da_arr[i])

        x1_plot = np.hstack(x1_plot)
        x2_plot = np.hstack(x2_plot)
        Da_plot = np.hstack(Da_plot)

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d', proj_type = 'ortho')

        tmp_planes = ax.zaxis._PLANES 
        ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                            tmp_planes[0], tmp_planes[1], 
                            tmp_planes[4], tmp_planes[5])

        ax.scatter(x1_plot, x2_plot, Da_plot, c='k', edgecolor='none', s=4)
        ax.set_xlabel('\n\n\nx1', fontsize=20)
        ax.set_ylabel('\n\n\nx2', fontsize=20)
        ax.set_zlabel('Da\n\n\n', fontsize=20)
        ax.set_title('Bifurcation diagram and Limit Cycles', fontsize=30)
        ax.tick_params(axis='both', labelsize=16, direction='in')
        ax.view_init(elev=30, azim=120)
        ax.dist = 10
        ax.grid(False)

    RHS_hair_error_trueLC_True = []
    RHS_hair_error_trueLC_Pred = []

    T = 1
    teval = np.linspace(0, T, 100)

    for i in range(x1_sample.size):
        Da_index = Da_sampler.choice(len(Da_arr))
        LC_index = LC_sampler.choice(np.array(x1LC[Da_index]).size)

        init = np.array([np.array(x1LC[Da_index]).reshape((-1))[LC_index],\
                         np.array(x2LC[Da_index]).reshape((-1))[LC_index]])
        pars_true = get_pars(Da_arr[Da_index])
        pars_ANN = np.array([Da_arr[Da_index]]).reshape((-1))

        sol_true = solve_ivp(f_cstr, y0=init, t_span=[0, T],
                                t_eval=teval, args=pars_true,
                                rtol=1e-5, atol=1e-8)

        sol_pred = solve_ivp(my_ode, y0=init, t_span=[0, T],
                                t_eval=teval, args=(pars_ANN,),
                                rtol=1e-5, atol=1e-8)

        RHS_hair_error_trueLC_True.append(sol_true.y[:,-1])
        RHS_hair_error_trueLC_Pred.append(sol_pred.y[:,-1])

        if make_plots:
            Da_plot = np.zeros(sol_true.y[0,:].shape) +  Da_arr[Da_index]
            ax.plot(sol_true.y[0,:],sol_true.y[1,:], Da_plot, 'k', label='True ODE', linewidth=2)
            ax.plot(sol_pred.y[0,:],sol_pred.y[1,:], Da_plot, 'b', label='Trained Model', linewidth=2)

        # if make_plots:

        #     axes[0].plot(sol_true.y[0,:],sol_true.y[1,:], 'k', label='True ODE', linewidth=2)
        #     axes[0].plot(sol_pred.y[0,:],sol_pred.y[1,:], 'b', label='Trained Model', linewidth=2)
        #     axes[0].set_ylabel('x  ', color='k', fontsize=35, rotation=0)
        #     axes[0].tick_params(axis='y', labelsize=20, direction='in', length=8, width=2)
        #     axes[0].set_xlabel('z', color='k', fontsize=35)
        #     axes[0].tick_params(axis='x', labelsize=20, direction='in', length=8, width=2)

        #     axes[1].plot(sol_true.y[5,:],sol_true.y[0,:], 'k', label='True ODE', linewidth=2)
        #     axes[1].plot(sol_pred.y[5,:],sol_pred.y[0,:], 'b', label='Trained Model', linewidth=2)
        #     axes[1].set_ylabel('x  ', color='k', fontsize=35, rotation=0)
        #     axes[1].tick_params(axis='y', labelsize=20, direction='in', length=8, width=2)
        #     axes[1].set_xlabel('g', color='k', fontsize=35)
        #     axes[1].tick_params(axis='x', labelsize=20, direction='in', length=8, width=2)

    RHS_hair_error_trueLC_Pred = np.array(RHS_hair_error_trueLC_Pred)
    RHS_hair_error_trueLC_True = np.array(RHS_hair_error_trueLC_True)

    scaler = MinMaxScaler()
    scaler.fit(RHS_hair_error_trueLC_True)
    
    RHS_hair_error_trueLC_Pred_norm = scaler.transform(RHS_hair_error_trueLC_Pred)
    RHS_hair_error_trueLC_True_norm = scaler.transform(RHS_hair_error_trueLC_True)

    L1_norm = np.sum(np.sum((np.abs(RHS_hair_error_trueLC_Pred_norm - RHS_hair_error_trueLC_True_norm)), axis=1), axis=0) / RHS_hair_error_trueLC_True_norm.size
    L2_norm = np.sum(np.sqrt(np.sum(((RHS_hair_error_trueLC_Pred_norm - RHS_hair_error_trueLC_True_norm)**2), axis=1)), axis=0) / RHS_hair_error_trueLC_True_norm.size
    Linf_norm = np.sum(np.max(np.abs(RHS_hair_error_trueLC_Pred_norm - RHS_hair_error_trueLC_True_norm), axis=1)) / RHS_hair_error_trueLC_True_norm.shape[0]

    return RHS_hair_error_trueLC_True, RHS_hair_error_trueLC_Pred, L1_norm, L2_norm, Linf_norm

def metric_RHS(filename=None, make_plots=False):
    Da_sample, x1_sample, x2_sample = cloud_sample()
    network = load_network(filename)

    x1_sample = np.array(x1_sample)
    x2_sample = np.array(x2_sample)
    Da_sample = np.array(Da_sample)

    x_in = torch.from_numpy(np.vstack((x1_sample,x2_sample)).T).to(network.device)
    p_in = torch.from_numpy(Da_sample).unsqueeze(-1).to(network.device)

    myB, mybeta, myD = 11, 3, np.array(Da_sample)
    real_RHS_x1 = -x1_sample + Da_sample * (1-x1_sample) * np.exp(x2_sample)
    real_RHS_x2 = -x2_sample + myB * Da_sample * (1-x1_sample) * np.exp(x2_sample) - mybeta * x2_sample

    ANN_output = network.output(x_in, p_in).detach().cpu().squeeze().numpy()
    pred_RHS_x1 = ANN_output[...,0]
    pred_RHS_x2 = ANN_output[...,1]

    real_RHS = np.stack((real_RHS_x1, real_RHS_x2), axis=1)
    pred_RHS = ANN_output.copy()

    scaler = MinMaxScaler()
    scaler.fit(real_RHS)
    
    real_RHS_norm = scaler.transform(real_RHS)
    pred_RHS_norm = scaler.transform(pred_RHS)

    L1_norm = np.sum(np.sum(np.abs(real_RHS_norm - pred_RHS_norm), axis=1)) / real_RHS_norm.size
    L2_norm = np.sum(np.sqrt(np.sum((real_RHS_norm - pred_RHS_norm) ** 2, axis=1))) / real_RHS_norm.size
    Linf_norm = np.sum(np.max(np.abs(real_RHS_norm - pred_RHS_norm), axis=1)) / real_RHS_norm.shape[0]

    if make_plots:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
        ax1.scatter(real_RHS_norm[:,0], pred_RHS_norm[:,0], s=1)
        ax1.plot([min(real_RHS_norm[:,0]), max(real_RHS_norm[:,0])], \
            [min(real_RHS_norm[:,0]), max(real_RHS_norm[:,0])], 'k-', lw=2) 
        ax1.set_xlabel('Real RHS x1', fontsize=15)
        ax1.set_ylabel('Predicted RHS x1', fontsize=15)
        ax1.xaxis.set_tick_params(labelsize=12, direction='in')
        ax1.yaxis.set_tick_params(labelsize=12, direction='in')


        ax2.scatter(real_RHS_norm[:,1], pred_RHS_norm[:,1], s=1)
        ax2.plot([min(real_RHS_norm[:,1]), max(real_RHS_norm[:,1])], \
            [min(real_RHS_norm[:,1]), max(real_RHS_norm[:,1])], 'k-', lw=2)
        ax2.set_xlabel('Real RHS x2', fontsize=15)
        ax2.set_ylabel('Predicted RHS x2', fontsize=15)
        ax2.xaxis.set_tick_params(labelsize=12, direction='in')
        ax2.yaxis.set_tick_params(labelsize=12, direction='in')

        fig.suptitle('RHS Prediction', fontsize=20)
        plt.show()


    return real_RHS_x1, real_RHS_x2, pred_RHS_x1, pred_RHS_x2, L1_norm, L2_norm, Linf_norm

def cloud_sample():
    
    unstable_ss, unstable_Da, stable_ss1, stable_Da1, stable_ss2, stable_Da2, Da_arr, x1min, x2min, x1max, x2max, x1LC, x2LC = \
    make_Bifurc_true_with_LC()

    Da_sampler = default_rng(seed=123)
    x1_sampler = default_rng(seed=234)
    x2_sampler = default_rng(seed=345)
    LC_sampler = default_rng(seed=456)

    Da_sample = np.zeros(10000)
    x1_sample = np.zeros(10000)
    x2_sample = np.zeros(10000)

    for i in range(x1_sample.size):
        Da_index = Da_sampler.choice(len(Da_arr))
        LC_index = LC_sampler.choice(np.array(x1LC[Da_index]).size)
        x1_sample[i] = np.array(x1LC[Da_index]).reshape((-1))[LC_index] * x1_sampler.uniform(low=0.98,high=1.02)
        x2_sample[i] = np.array(x2LC[Da_index]).reshape((-1))[LC_index] * x2_sampler.uniform(low=0.98,high=1.02)
        Da_sample[i] = Da_arr[Da_index]

    return Da_sample, x1_sample, x2_sample

def load_network(filename=None):
    
#     xmaxmin = np.savez("minmax/minmax.npz",xmax, xmin)
    
    f = np.load("/home/smalani/PartialObservations_BF/PartialObservations/minmax/minmax.npz")
    
    xmax = f['arr_0']
    xmin = f['arr_1']
    # print(xmax)
    # print(xmin)
    
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

                def raw_output(self, x, par):

                    ANN_input = torch.cat((self.norm_func(x), par), dim=-1)
                    g = self.net(ANN_input)[...,0]
                    return g
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

                def raw_output(self, x, par):

                    ANN_input = torch.cat((self.norm_func(x), par), dim=-1)
                    g = self.net(ANN_input)[...,0]
                    return g
        else:
            raise ValueError("Tell me whether to train the parameters!")
    else:
        raise ValueError("Tell me what box to use!")
    network = my_Network(mlp, config["DATA"]["N_TRAIN"], 2, norm_func=norm_func, inv_norm_func=inv_norm_func, 
                      init_available=config["DATA"]["INIT_AVAILABLE"], integrator='RK4')
    device = 'cpu'

    # filename = config["DATA"]["PATH"]+'model_' + '.net'
    if filename is None:
        filename = "/home/smalani/PartialObservations_BF/PartialObservations/data/"+'model_' + '.net'

    print(filename)

    state_dict = torch.load(filename, map_location=torch.device(device))

    network.load_state_dict(state_dict, strict=False)

    # print(network)
#     network.to(device)

    network.double()
    return network


##################################################################

##################################################################



def make_Bifurc_true(make_plots=False):
    Da_arr = np.linspace(0.2,0.5,100)
    
    stable_ss1 = []
    stable_Da1 = []
    
    stable_ss2 = []
    stable_Da2 = []
    
    unstable_ss = []
    unstable_Da = []
    
    x1max = []
    x1min = []

    period = []
    period_point = []
    
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
            period.append(y[1])
            period_point.append(y0_int)
                    
        else:
            if not switch:
                stable_ss1.append(root)
                stable_Da1.append(Da)
                
                x1max.append(root[0])
                x2max.append(root[1])
                x1min.append(root[0])
                x2min.append(root[1])
                period.append(0)
                period_point.append(root)
            else:
                stable_ss2.append(root)
                stable_Da2.append(Da)
                
                x1max.append(root[0])
                x2max.append(root[1])
                x1min.append(root[0])
                x2min.append(root[1])
                period.append(0)
                period_point.append(root)
        
        roots.append(root)
    roots = np.array(roots)
    unstable_ss = np.array(unstable_ss)
    unstable_Da = np.array(unstable_Da)
    stable_ss1 = np.array(stable_ss1)
    stable_Da1 = np.array(stable_Da1)
    stable_ss2 = np.array(stable_ss2)
    stable_Da2 = np.array(stable_Da2)
    period = np.array(period)
    period_point = np.array(period_point)
    
    if make_plots:
    
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
        fig.savefig('/home/smalani/PartialObservations_BF/PartialObservations/data/Bifurcation Plot True x1.png',format='png')

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
        fig.savefig('/home/smalani/PartialObservations_BF/PartialObservations/data/Bifurcation Plot True x2.png',format='png')
    return unstable_ss, unstable_Da, stable_ss1, stable_Da1, stable_ss2, stable_Da2, Da_arr, x1min, x2min, x1max, x2max, period, period_point

def make_Bifurc_prediction(make_plots=False, filename=None):
    network = load_network(filename)
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
    
    if make_plots:

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
 
def make_Bifurc_true_with_LC(make_plots=False, B=11, beta=3):
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

    x1LC = []
    x2LC = []
    
    roots = []
    switch = False
    root = [0.5, 3]
    for i in range(len(Da_arr)):
        Da = Da_arr[i]
        
        pars = get_pars(Da, B=B, beta=beta)
        
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

            x1LC.append(sol.y[0,::10])
            x2LC.append(sol.y[1,::10])
                    
        else:
            if not switch:
                stable_ss1.append(root)
                stable_Da1.append(Da)
                
                x1max.append(root[0])
                x2max.append(root[1])
                x1min.append(root[0])
                x2min.append(root[1])

                x1LC.append(root[0])
                x2LC.append(root[1])
            else:
                stable_ss2.append(root)
                stable_Da2.append(Da)
                
                x1max.append(root[0])
                x2max.append(root[1])
                x1min.append(root[0])
                x2min.append(root[1])

                x1LC.append(root[0])
                x2LC.append(root[1])
        
        roots.append(root)
    roots = np.array(roots)
    unstable_ss = np.array(unstable_ss)
    unstable_Da = np.array(unstable_Da)
    stable_ss1 = np.array(stable_ss1)
    stable_Da1 = np.array(stable_Da1)
    stable_ss2 = np.array(stable_ss2)
    stable_Da2 = np.array(stable_Da2)
    
    if make_plots:
    
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
        # fig.savefig('/home/smalani/PartialObservations/data/Bifurcation Plot True x1.png',format='png')

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
        # fig.savefig('/home/smalani/PartialObservations/data/Bifurcation Plot True x2.png',format='png')

        x1_plot = []
        x2_plot = []
        Da_plot = []

        for i in range(len(x1LC)):
            x1_plot.append(x1LC[i])
            x2_plot.append(x2LC[i])
            Da_plot.append(np.zeros(np.array(x1LC[i]).shape) +  Da_arr[i])

        x1_plot = np.hstack(x1_plot)
        x2_plot = np.hstack(x2_plot)
        Da_plot = np.hstack(Da_plot)


        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d', proj_type = 'ortho')

        tmp_planes = ax.zaxis._PLANES 
        ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                            tmp_planes[0], tmp_planes[1], 
                            tmp_planes[4], tmp_planes[5])

        ax.scatter(x1_plot, x2_plot, Da_plot, c=Da_plot, cmap='viridis', edgecolor='none', s=10)
        ax.set_xlabel('\n\n\nx1', fontsize=20)
        ax.set_ylabel('\n\n\nx2', fontsize=20)
        ax.set_zlabel('Da\n\n\n', fontsize=20)
        ax.set_title('Bifurcation diagram and Limit Cycles', fontsize=30)
        ax.tick_params(axis='both', labelsize=16, direction='in')
        ax.view_init(elev=30, azim=120)
        ax.dist = 10
        ax.grid(False)
        plt.show()
    return unstable_ss, unstable_Da, stable_ss1, stable_Da1, stable_ss2, stable_Da2, Da_arr, x1min, x2min, x1max, x2max, x1LC, x2LC
    
def myf_cstr_torch(t, y, Da, beta, B):
    """Time derivatives of the CSTR model."""
    x1, x2 = y
    dx1dt = -x1 + Da * (1-x1) * torch.exp(x2)
    dx2dt = -x2 + B * Da * (1-x1) * torch.exp(x2) - beta * x2

    return torch.stack((dx1dt, dx2dt))

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