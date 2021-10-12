import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from tqdm.auto import tqdm

from config import config

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
                  full_observations=True,
                  Da_random=False,
                  detail=False,
                 ):
    
    if Da_random:
        Da_base = np.random.uniform(0.2,0.5,10)   
    else:
        Da_base = np.linspace(0.2,0.5,10)
    Da = np.random.choice(Da_base, n_train, replace=True)

    if reg_time:
        x1_times = np.linspace(skip_time, tmax+skip_time, x1_sample_num).round(decimals=5)
        x2_times = np.linspace(skip_time, tmax+skip_time, x1_sample_num).round(decimals=5)
    else:
        x1_times = np.sort(np.random.uniform(skip_time, tmax+skip_time, x1_sample_num)).round(decimals=5)
        if full_observations: x2_times = x1_times
        else: x2_times = np.sort(np.random.uniform(skip_time, tmax+skip_time, x1_sample_num)).round(decimals=5)
        
            
    solver_times = np.sort(np.unique(np.concatenate(([skip_time], x1_times, x2_times))))
    
    output = np.zeros((n_train,2), dtype=object)
    if detail:
        output_detail = np.zeros((n_train,2,1000), dtype=object)
        output_detail_t = np.zeros((n_train,1000), dtype=object)
    
    for i in tqdm(range(n_train), leave=True, position=0):
        sol = integrate_cstr(tmin=0, tmax=tmax+skip_time+0.1, Da=Da[i], B=B, beta=beta, teval=solver_times)
            
        # Extra copy of solution at higher time resolution for better plotting of 'true' trajectory        
        output[i,0] = (sol.y[0, np.in1d(sol.t.round(decimals=5),x1_times)], x1_times - skip_time)
        output[i,1] = (sol.y[1, np.in1d(sol.t.round(decimals=5),x2_times)], x2_times - skip_time)
        
        if detail:
            sol_detail = integrate_cstr(tmin=skip_time, tmax=tmax+skip_time, Da=Da[i], B=B, beta=beta, teval=np.linspace(skip_time,tmax+skip_time,1000), y0 = sol.y[:,0])

            output_detail[i,0,:] = sol_detail.y[0,:]
            output_detail[i,1,:] = sol_detail.y[1,:]
            output_detail_t[i,:] = sol_detail.t - skip_time
        
    if detail:
        return Da, output, output_detail_t, output_detail
    else:
        return Da, output