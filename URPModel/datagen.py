import sys
sys.path.append("..")

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm
from config import config

np.random.seed(1234)

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


def integrate_cstr(tmin=0, tmax=20, y0=None, verbose=False, Da=0.25, B=11, beta=3, teval=np.linspace(0, 20, 2001)):

    pars = get_pars(Da,B,beta)
    if y0 is None:
        y0 = cstr_initial_conditions()

    sol = solve_ivp(f_cstr, y0=y0, t_span=[tmin, tmax],
                    t_eval=teval, args=pars,
                    rtol=1e-7, atol=1e-10, max_step=0.01)
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

def generate_data(n_train=config["DATA"]["N_TRAIN"], 
                  tmax=config["DATA"]["TMAX"], 
                  B=config["PAR"]["B"], 
                  beta=config["PAR"]["beta"],                   
                  x1_sample_num=config["DATA"]["X1_SAMPLE_NUM"],
                  x2_sample_num=config["DATA"]["X2_SAMPLE_NUM"], 
                  skip_time=config["DATA"]["SKIP_TIME"],
                  reg_time=config["DATA"]["REG_TIME"],
                  dt_dist=(config["DATA"]["DT_MU"],
                           config["DATA"]["DT_SIGMA"]),
                  init_available=config["DATA"]["INIT_AVAILABLE"],
                  full_observations=config["DATA"]["FULL_OBSERVATIONS"],
                  Da_random=False,
                  detail=False,
                  Da_set=0.33,
                 ):
    
    """ 
    Outputs parameter array of shape (n x dp). Each element of this object array is a tuple (p,t).
    p and t are numpy arrays of shape (Tp,). 
    p is array of time steps of parameter change, and t the corresponding times.
    Parameter values are assumed to be constant until next specified update time
    
    Outputs data object array of shape (n x dx). Each element of this object array is a tuple (x, t). 
    x and t are numpy arrays of shape (Tx,). 
    x is array of observations of the variable, and t the corresponding times for each of the observations. 
    
    Optionally outputs high temporal resolution t (shape n x T=1000 x dx) and x (shape n x T=1000) numpy arrays for plotting of ground truth trajectories.
    """
    Da_rng = np.random.default_rng(seed = 2341)
    dt_rng = np.random.default_rng(seed = 3412)
    
    if n_train == 1:
        Da = [Da_set]
    else:
        if Da_random:
            Da_base = Da_rng.uniform(0.2,0.5,10)   
        else:
            Da_base = np.linspace(0.2,0.5,10)
        Da = Da_rng.choice(Da_base, n_train, replace=True)
    
    solver_times_arr = []
    x1_times_arr = []
    x2_times_arr = []
    
    for i in range(n_train):
        if reg_time:
            x1_times = np.linspace(skip_time, tmax+skip_time, x1_sample_num)
            x2_times = np.linspace(skip_time, tmax+skip_time, x2_sample_num)
            x1_times_arr.append(np.array(x1_times).round(decimals=5))
            x2_times_arr.append(np.array(x2_times).round(decimals=5))
        else:
            mu, sigma = dt_dist
            x1_times, x2_times = [], []
            if init_available:
                k = (mu/sigma) ** 2
                theta = mu/k
                x1_times.append(skip_time)
                x2_times.append(skip_time)
            else:
                mu = mu * (max(x1_sample_num, x2_sample_num) - 1) / max(x1_sample_num, x2_sample_num)
                k = (mu/sigma) ** 2
                theta = mu/k
                if full_observations:
                    x1x2_dt = np.abs(dt_rng.gamma(k, theta))
                    x1_times.append(skip_time + x1x2_dt)
                    x2_times.append(skip_time + x1x2_dt)
                else:
                    x1_times.append(skip_time + np.abs(dt_rng.gamma(k, theta)))
                    x2_times.append(skip_time + np.abs(dt_rng.gamma(k, theta)))

            if full_observations:
                assert x1_sample_num == x2_sample_num
                while len(x1_times) < x1_sample_num:
                    x1x2_dt = np.abs(dt_rng.gamma(k, theta))
                    x1_times.append(x1_times[-1] + x1x2_dt)
                    x2_times.append(x2_times[-1] + x1x2_dt)
            else:
                while len(x1_times) < x1_sample_num:
                    x1_times.append(x1_times[-1] + np.abs(dt_rng.gamma(k, theta)))
                while len(x2_times) < x2_sample_num:
                    x2_times.append(x2_times[-1] + np.abs(dt_rng.gamma(k, theta)))
            x1_times_arr.append(np.array(x1_times).round(decimals=5))
            x2_times_arr.append(np.array(x2_times).round(decimals=5))
        solver_times_arr.append(np.sort(np.unique(np.concatenate(([skip_time], x1_times_arr[i], x2_times_arr[i])))))
    
    output = np.zeros((n_train,2), dtype=object)
    Da_output = np.zeros((n_train,1), dtype=object)
    if detail:
        output_detail = np.zeros((n_train,1000,2))
        output_detail_t = np.zeros((n_train,1000))
        output_detail_Da = np.zeros((n_train,1000))
    
    for i in tqdm(range(n_train), leave=True, position=0):
        sol = integrate_cstr(tmin=0, tmax=max(solver_times_arr[i]), Da=Da[i], B=B, beta=beta, teval=solver_times_arr[i])
            
        # Extra copy of solution at higher time resolution for better plotting of 'true' trajectory        
        output[i,0] = (sol.y[0, np.in1d(sol.t.round(decimals=5),x1_times_arr[i])], x1_times_arr[i] - skip_time)
        output[i,1] = (sol.y[1, np.in1d(sol.t.round(decimals=5),x2_times_arr[i])], x2_times_arr[i] - skip_time)
        Da_output[i,0] = (np.array([Da[i]]), np.array([skip_time]) - skip_time)
        
        if detail:
            sol_detail = integrate_cstr(tmin=skip_time, tmax=max(solver_times_arr[i]), Da=Da[i], B=B, beta=beta, teval=np.linspace(skip_time,max(solver_times_arr[i]),1000), y0 = sol.y[:,0])

            output_detail[i,:,0] = sol_detail.y[0,:]
            output_detail[i,:,1] = sol_detail.y[1,:]
            output_detail_t[i,:] = sol_detail.t - skip_time
            output_detail_Da[i,:] = np.array([Da[i]])
        
    if detail:
        return output, Da_output, output_detail_t, output_detail, output_detail_Da
    else:
        return output, Da_output