import sys
sys.path.append("..")

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm
from PartialObservations.config import config

import matplotlib.pyplot as plt

np.random.seed(1234)

def initial_conditions(ic='random'):
    if ic == 'random':
        # myvar0 = np.concatenate((np.random.uniform(0.01,100,1),
        #                  np.random.uniform(0.01,1.5,1),
        #                  np.random.uniform(0.01,1.5,1),
        #                  np.random.uniform(0.01,250,1),
        #                  np.random.uniform(0.01,2200,1),
        #                  np.random.uniform(0.01,4.5,1)))

        f = np.load('minmax/initialcond.npz')
        myvar0 = f['arr_0']
        index = np.random.choice(myvar0.shape[1])
        range = (np.max(myvar0, axis=1) - np.min(myvar0, axis=1))[0,:]
        mean = ((np.max(myvar0, axis=1) - np.min(myvar0, axis=1)) / 2 + np.min(myvar0, axis=1))[0,:]
        init = myvar0[0, index, :]
        loc = (mean - init) / 2
        # loc[1:] = 0

        loc[1] = 0.2
        range[1] = 1

        init = np.abs(init + np.random.normal(loc=loc, scale=range/10))
        # init[1] = 0.

        # myvar0 = np.concatenate((50 * (1 + np.random.uniform(size=1)),
        #                          0.75 * (1 + np.random.uniform(size=1)),
        #                          0.75 * (1 + np.random.uniform(size=1)),
        #                          125 * (1 + np.random.uniform(size=1)),
        #                          1100 * (1 + np.random.uniform(size=1)),
        #                          2.25 * (1 + np.random.uniform(size=1))))
        return init
    else:
        # Not implemented
        assert False


def ode_fun(t, var, par):
    x, y, z, u, v, g = var
    alpha, uf, omega, sigma, rho, eta, phi1, phi2, uc1_prime, uc2_prime, uc3_prime = par

    u1_prime = u / ((1+u) * (1+g))
    u2_prime = phi1 * v / (1+v)
    u3_prime = phi2 * v / (sigma + v)

    # print('my ode fun')
    # print(par)
    # assert False

    output = []
    for i in range(6):
        output.append([])
    
    output[0] = -alpha * x + u1_prime * x - uc1_prime * x
    output[1] = -alpha * y + u2_prime * y - uc2_prime * y
    output[2] = -alpha * z + u3_prime * z - uc3_prime * z
    output[3] = alpha * (uf - u) - u1_prime * x
    output[4] = -alpha * v + omega * u1_prime * x - u2_prime * y - sigma * u3_prime * z
    output[5] = -alpha * g + rho * u2_prime * y + eta * u3_prime * z

    return np.squeeze(np.stack(output, axis=-1))

def torch_ode_fun(t, var, par):
    x, y, z, u, v, g = var
    alpha, uf, omega, sigma, rho, eta, phi1, phi2, uc1_prime, uc2_prime, uc3_prime = par

    u1_prime = u / ((1+u) * (1+g))
    u2_prime = phi1 * v / (1+v)
    u3_prime = phi2 * v / (sigma + v)

    output = torch.zeros(6)
    
    output[0] = -alpha * x + u1_prime * x - uc1_prime * x
    output[1] = -alpha * y + u2_prime * y - uc2_prime * y
    output[2] = -alpha * z + u3_prime * z - uc3_prime * z
    output[3] = alpha * (uf - u) - u1_prime * x
    output[4] = -alpha * v + omega * u1_prime * x - u2_prime * y - sigma * u3_prime * z
    output[5] = -alpha * g + rho * u2_prime * y + eta * u3_prime * z

    return output



def par_fun(u_m1=0.68, u_m2=0.20, u_m3=0.25, u_c1=0.25, u_c2=0.08, u_c3=0.11,
            K_1=0.01, K_2=0.001, K_3=0.01, K_i=0.01, Y_1=1/2, Y_2=1/1.37, Y_3=1/1.2,
            beta=1.94, gamma=1.8, delta=1.55, D=1/7.3, sf=2.5):
    
    alpha       =   D / u_m1
    uf          =   sf / K_1
    omega       =   beta * Y_1 * K_1 / K_2
    sigma       =   K_3 / K_2
    rho         =   gamma * Y_2 * K_2 / K_i
    eta         =   delta * Y_3 * K_3 / K_i
    phi1        =   u_m2 / u_m1
    phi2        =   u_m3 / u_m1
    uc1_prime   =   u_c1 / u_m1
    uc2_prime   =   u_c2 / u_m1
    uc3_prime   =   u_c3 / u_m1

    return alpha, uf, omega, sigma, rho, eta, phi1, phi2, uc1_prime, uc2_prime, uc3_prime
    


# def testme():
#     mypar = par_fun(D=1/7.3, sf=2.5)
#     t = 0

#     print(mypar)

#     # myvar0 = np.array([2.45524443e+01, 9.40042585e-11, 1.94133806e-01, 1.96344728e+02,
#     #  5.16882041e+02, 4.61654478e-01])
#     myvar0 = [7.54651217e+001, 5.68061229e-110, 3.16832990e-001, 3.07716331e+001,
#     2.12083688e+003, 7.33444520e-001]

#     myvar0 = [7.83577452e+01, 1.76910142e-01, 1.21884731e-02, 1.24296145e+00,
#     2.41269019e+03, 3.27558469e-02]

#     myvar0 = np.concatenate((np.random.uniform(25,75,1),
#                             np.random.uniform(0,0.0001,1),
#                             np.random.uniform(0.28, 0.35,1),
#                             np.random.uniform(70, 200,1),
#                             np.random.uniform(800,1900,1),
#                             np.random.uniform(0.6,0.8,1)))

#     # zero_sol = fsolve(lambda y, par: ode_fun(0,y,par), myvar0, args=(mypar,))
#     # print(zero_sol)

#     tspan = [0,500]
#     tskip = 500

#     sol = solve_ivp(ode_fun, y0=myvar0, 
#                     t_span=[tspan[0],tspan[1]+tskip], t_eval=np.linspace(tskip,tspan[1]+tskip,1000), args=(mypar,), 
#                     rtol=1e-5, atol=1e-8)

#     labels = ['x', 'y', 'z', 'u', 'v', 'g']

#     print(sol.y[:,-1])

#     plt.figure(figsize=(20,10))

#     for i in range(6):
#         plt.subplot(int(str(61) + str(i+1)))
#         plt.plot(sol.t, sol.y[i,:])
#         plt.ylabel(labels[i])
#     plt.xlabel(r'theta')
#     plt.show()

#     return sol



def integrate_cstr(tmin=0, tmax=20, y0=None, verbose=False, theta=7.3, sf=2.5, teval=None):
    if teval is None:
        teval = np.linspace(tmin, tmax, 2001)
    pars = par_fun(D=1/theta, sf=sf)

    if y0 is None:
        y0 = initial_conditions()


    # assert False

    sol = solve_ivp(ode_fun, y0=y0, t_span=[tmin, tmax],
                    t_eval=teval, args=(pars,),
                    rtol=1e-5, atol=1e-8)
    if verbose:
        print(sol.message)

    if verbose:
        labels = ['x', 'y', 'z', 'u', 'v', 'g']

        plt.figure(figsize=(20,10))

        for i in range(6):
            plt.subplot(int(str(61) + str(i+1)))
            plt.plot(sol.t, sol.y[i,:])
            plt.ylabel(labels[i])
        plt.xlabel(r'theta')

    return sol

def generate_data(n_train=config["DATA"]["N_TRAIN"], 
                  tmax=config["DATA"]["TMAX"], 
                  sf=config["PAR"]["sf"],                   
                  x_sample_num=config["DATA"]["X_SAMPLE_NUM"],
                  skip_time=config["DATA"]["SKIP_TIME"],
                  reg_time=config["DATA"]["REG_TIME"],
                  dt_dist=(config["DATA"]["DT_MU"],
                           config["DATA"]["DT_SIGMA"]),
                  init_available=config["DATA"]["INIT_AVAILABLE"],
                  full_observations=True,
                  theta_random=False,
                  detail=False,
                  theta_set=7.3,
                  duplicate_reverse=False
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
    theta_rng = np.random.default_rng(seed = 2341)
    dt_rng = np.random.default_rng(seed = 3412)

    if n_train == 1:
        theta = [theta_set]
    else:
        if theta_random:
            # theta_base = theta_rng.uniform(7.,9.,10)
            theta_base = theta_rng.uniform(7.3,7.3,10)  
        else:
            # theta_base = np.linspace(7.,9.,10) 
            theta_base = np.linspace(7.3,7.3,10)  
        theta = theta_rng.choice(theta_base, n_train, replace=True)
    
    solver_times_arr = []
    x_times_arr = []
    for _ in range(n_train):
        x_times_arr.append([])
    
    for i in range(n_train):
        for _ in range(len(x_sample_num)):
            x_times_arr[i].append([])
        if reg_time:
            for j in range(len(x_sample_num)):
                x_times = np.linspace(skip_time, tmax+skip_time, x_sample_num[j]).round(decimals=5)
                x_times_arr[i][j] = x_times
        else:
            mu, sigma = dt_dist
            x_times = []
            for _ in range(len(x_sample_num)):
                x_times.append([])
            if init_available:
                k = (mu/sigma) ** 2
                theta_dist = mu/k
                for j in range(len(x_sample_num)):
                    x_times[j].append(skip_time)
            else:
                mu = mu * (max(x_sample_num) - 1) / max(x_sample_num)
                k = (mu/sigma) ** 2
                theta_dist = mu/k
                for j in range(len(x_sample_num)):
                    x_times[j].append(skip_time + np.abs(dt_rng.gamma(k, theta_dist)))
            for j in range(len(x_sample_num)):
                while len(x_times[j]) < x_sample_num[j]:
                    x_times[j].append(x_times[j][-1] + np.abs(dt_rng.gamma(k, theta_dist)))
            for j in range(len(x_sample_num)):
                x_times_arr[i][j] = np.array(x_times[j]).round(decimals=5)

        solver_times_arr.append(np.unique(np.concatenate(([skip_time], *x_times_arr[i]))))

    
    # print('bloopybloop')
    # print(len(x_times_arr))
    # print(len(x_times_arr[0]))
    # print(x_times_arr[0][0].shape)
    # assert False

    output = np.zeros((n_train,len(x_sample_num)), dtype=object)
    theta_output = np.zeros((n_train,1), dtype=object)
    if detail:
        output_detail = np.zeros((n_train,1000,len(x_sample_num)))
        output_detail_t = np.zeros((n_train,1000))
        output_detail_theta = np.zeros((n_train,1000))

    for i in tqdm(range(n_train), leave=True, position=0):
        sol = integrate_cstr(tmin=0, tmax=max(solver_times_arr[i]), theta=theta[i], sf=sf, teval=solver_times_arr[i])
            
        
        for j in range(len(x_sample_num)):   
            output[i,j] = (sol.y[j, np.in1d(sol.t.round(decimals=5).astype(float),x_times_arr[i][j])], x_times_arr[i][j] - skip_time)
        theta_output[i,0] = (np.array([theta[i]]), np.array([skip_time]) - skip_time)
        
        # Extra copy of solution at higher time resolution for better plotting of 'true' trajectory  
        if detail:
            sol_detail = integrate_cstr(tmin=skip_time, tmax=max(solver_times_arr[i]), theta=theta[i], sf=sf, teval=np.linspace(skip_time,max(solver_times_arr[i]),1000), y0 = sol.y[:,0])

            for j in range(len(x_sample_num)): 
                output_detail[i,:,j] = sol_detail.y[j,:].astype(float)
            output_detail_t[i,:] = sol_detail.t - skip_time
            output_detail_theta[i,:] = np.array([theta[i]])


    # if duplicate_reverse:
    #     output_dup = np.zeros((n_train,len(x_sample_num)), dtype=object)
    #     theta_output_dup = np.zeros((n_train,1), dtype=object)
    #     for i in range(n_train):
    #         for j in range(len(x_sample_num)):  
    #             output_dup[i,j] = (np.flip(output[i,j][0].copy()), np.flip(output[i,j][1].copy()))
    #         theta_output_dup[i,0] = (np.flip(theta_output[i,0][0].copy()), np.flip(theta_output[i,0][1].copy()))
        
    #     # print('output shapes')
    #     # print(output.shape)
    #     output = np.concatenate((output, output_dup), axis=0)
    #     # print(output.shape)
    #     # assert False
    #     theta_output = np.concatenate((theta_output, theta_output_dup), axis=0)


    if detail:
        print('pakapaaakaaa detail')  
        print(output.shape)
        return output, theta_output, output_detail_t, output_detail, output_detail_theta
    else:
        print('pakapaaakaaa')  
        print(output.shape)
        return output, theta_output