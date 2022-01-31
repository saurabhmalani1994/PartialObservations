import numpy as np

config = {}
config["DATA"] = {}
config["DATA"]["TMAX"] = 200
# config["DATA"]["L_TRAJECTORIES"] = 200 # 200 equal lengths is dt = 0.02
config["DATA"]["N_TRAIN"] = 200 # Num of training traj
config["DATA"]["N_VAL"] = 20
config["DATA"]["N_TEST"] = 1
config["DATA"]["PATH"] = '../data/'
config["DATA"]["SKIP_TIME"] = 0
config["DATA"]["X_SAMPLE_NUM"] = (np.array([50, 50, 50, 50, 50, 50]) / 2).astype(int) # Num of pts per traj
config["DATA"]["MAX_DELTA_T"] = 5 # Timestep in integrator
config["DATA"]["DT_MU"] = config["DATA"]["TMAX"] / (config["DATA"]["X_SAMPLE_NUM"][0] - 1)
config["DATA"]["DT_SIGMA"] = config["DATA"]["DT_MU"] / 2
config["DATA"]["REG_TIME"] = False   # True for regular sampling in time (Delta_T = const), False for random sampling in time Uniform~ (0, Delta_T)
config["DATA"]["INIT_AVAILABLE"] = True   # True for ALL initial conditions available at time 0, False for Not available
config["DATA"]["FULL_OBSERVATIONS"] = False   # Only activates for REG_TIME = False ## NOT IMPLEMENTED FOR BandF YET
config["DATA"]["DUP_REVERSE"] = False   # To train forwards AND backwards in time!

config["PAR"] = {}
config["PAR"]["sf"] = 2.5

config["TRAINING"] = {}
config["TRAINING"]["BATCH_SIZE"] = 512
config["TRAINING"]["LEARNING_RATE"] = 1e-2
config["TRAINING"]["EPOCHS"] = 10000 # 1000 # 1259 # 2539 # 5260

config["MODEL"] = {}
config["MODEL"]["NUM_HIDDEN"] = [32, 32]
config["MODEL"]["BOX"] = 'Grey'
config["MODEL"]["Parameters"] = 'Fixed'