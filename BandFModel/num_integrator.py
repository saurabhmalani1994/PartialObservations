import numpy as np
from torchdiffeq import odeint
import torch

def torch_odeint(func, tspan, init, pars, dt):
    t_arr = np.arange(*tspan, step=dt)
    myfunc = lambda t, y: func(t,y,pars)
    x_out = odeint(myfunc, torch.tensor(init), torch.tensor(t_arr))

    return t_arr, x_out.detach().numpy().T

def RK4_int(func, tspan, init, pars, dt):

    t_arr = np.arange(*tspan, step=dt)
    x_out = []
    x_out.append(init)

    x_in = init

    for i, t in enumerate(t_arr[1:]):
        k1 = func(t, x_in, pars)
        x_in = x_out[i] + k1 * dt / 2

        k2 = func(t + dt/2, x_in, pars)
        x_in = x_out[i] + k2 * dt / 2

        k3 = func(t + dt/2, x_in, pars)
        x_in = x_out[i] + k3 * dt

        k4 = func(t + dt, x_in, pars)
        x_in = x_out[i] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        x_out.append(x_in)
    
    x_out = np.array(x_out)

    return t_arr, x_out.T

def DOPRI_int(func, tspan, init, pars, dt):

    t_arr = np.arange(*tspan, step=dt)
    x_out = []
    x_out.append(init)

    x_in = init

    for i, t in enumerate(t_arr[1:]):
        k1 = func(t, x_in, pars)
        x_in = x_out[i] + k1*dt*(1/5)

        k2 = func(t + dt*(1/5), x_in, pars)
        x_in = x_out[i] + k1*dt*(3/40) + k2*dt*(9/40)

        k3 = func(t + dt*(3/10), x_in, pars)
        x_in = x_out[i] + k1*dt*(44/45) - k2*dt*(56/15) \
                        + k3*dt*(32/9)

        k4 = func(t + dt*(4/5), x_in, pars)
        x_in = x_out[i] + k1*dt*(19372/6561) - k2*dt*(25360/2187)\
                        + k3*dt*(64448/6561) - k4*dt*(212/729)

        k5 = func(t + dt*(8/9), x_in, pars)
        x_in = x_out[i] + k1*dt*(9017/3168) - k2*dt*(355/33) \
                        + k3*dt*(46732/5247) + k4*dt*(49/176)\
                        - k5*dt*(5103/18656)

        k6 = func(t + dt, x_in, pars)
        x_in = x_out[i] + k1*dt*(35/384) + k2*dt*(0) \
                        + k3*dt*(500/1113) + k4*dt*(125/192) \
                        - k5*dt*(2187/6784) + k6*dt*(11/84)

        x_out.append(x_in)
    
    x_out = np.array(x_out)

    return t_arr, x_out.T