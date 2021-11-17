import numpy as np
import scipy as sp
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import torch


def periodic_sol(func, torch_func, pars, zero_init, period_init, Tguess, coord=0):
    sol = root(lambda y, par: func(0,y,par), np.abs(zero_init), args=(pars,),
                    method='hybr',
                     options={'factor': 0.1})
    zero_sol = sol.x
    print('zero_sol')
    print(zero_sol)

    torch_f_cstr = lambda y: torch_func(0, y, pars)

    J = torch.autograd.functional.jacobian(lambda y: torch_func(0, y, pars), torch.from_numpy(zero_sol), strict=True)
    w, v = np.linalg.eig(J)

    if ~(np.any(w.real>0)):
        return True, zero_sol, zero_sol, 0.
    
    if period_init is None:
        period_init = np.delete(zero_sol,coord)
        period_init = np.append(period_init, np.sqrt(Tguess - 0.1))
    else:
        period_init = np.delete(period_init,coord)
        period_init = np.append(period_init, np.sqrt(Tguess - 0.1))
    
    period_init = np.delete(zero_sol,coord)
    period_init = np.append(period_init, np.sqrt(Tguess - 0.1))

    # period_init = zero_sol.copy()
    # period_init[coord] = period_init[coord]*1.1

    # sol = solve_ivp(func, y0=period_init, t_span=[0, 200],
    #                     args=(pars,), rtol=1e-5, atol=1e-8)

    # period_init = sol.y[:,-1]
    # period_init = np.delete(period_init,coord)
    # period_init = np.append(period_init, np.sqrt(Tguess - 0.1))

    scale_fac = np.ones(zero_sol.shape)
    # scale_fac = 1/zero_sol
    scale_fac[period_init<1e-10] = 1e-30

    # print(scale_fac)
    # assert False
    print('Period Init')
    print(period_init)

    not_done = True
    mult = 0.1

    while not_done:
        
        # coord = 0
        # while zero_sol[coord] < 1e-10:
        #     coord += 1
        # print(coord)
        print(mult)

        sol = root(ODE_Bifurc, period_init, args=(func,pars,np.abs(zero_sol),coord,mult),
                    method='hybr',
                     options={'diag': scale_fac, 'factor': 0.1})
        mult = mult * (-0.8)
        period_init = sol.x
        # if mult < 1e-3:
        #     coord += 1
        #     while zero_sol[coord] < 1e-10:
        #         coord += 1
        #     mult = 0.1
        if sol.success is True:
            not_done = False
    period_out = sol.x

    # print(sol.message)
    # print('Period Out')
    # print(period_out)

    # period_out = least_squares(ODE_Bifurc, x_init, bounds = ((0, 0, 0, 0, 0, 10), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)),
    #                              args=(func,pars,zero_sol,coord))

    # print('Period Out LS')
    # print(period_out.x)
    # print(period_out.cost)
    # print(period_out.fun)
    # assert False

    output = period_out[:-1]
    output = np.insert(output, coord, zero_sol[coord] * (1+mult))

    T_period = period_out[-1]**2 + 0.1

    return False, zero_sol, output, T_period

def ODE_Bifurc(y, func, pars, init, coord=0, mult = 0.2):
    print(y)
    x = y[:-1]
    T = y[-1]
    print(T**2 + 0.1)
    y0 = np.insert(x, coord, init[coord] * (1+mult))

    
    sol = solve_ivp(func, y0=y0, t_span=[0, 0.1 + T**2],
                    args=(pars,), rtol=1e-5, atol=1e-8)
    
    # y_init = sol.y[:,-1]

    # sol = solve_ivp(func, y0=y_init, t_span=[0.1, T],
    #                 args=(pars,), rtol=1e-5, atol=1e-8)
    
    T_out = sol.t[-1]
    output = sol.y[:,-1] - y0
    print(output)
    print('-----------------')
    return output


def ODE_Bifurc_ls(y, func, pars, init, coord=0):
    x = y[:-1]
    T = y[-1]
    y0 = np.insert(x, coord, init[coord] * 0.95)

    
    sol = solve_ivp(func, y0=y0, t_span=[0, T],
                    args=(pars,), rtol=1e-5, atol=1e-8)
    
    # y_init = sol.y[:,-1]

    # sol = solve_ivp(func, y0=y_init, t_span=[0.1, T],
    #                 args=(pars,), rtol=1e-5, atol=1e-8)
    
    T_out = sol.t[-1]
    output = sol.y[:,-1] - y0

    return np.sqrt(np.sum(output ** 2))