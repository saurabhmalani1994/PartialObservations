config = {}
config["DATA"] = {}
config["DATA"]["TMAX"] = 6
# config["DATA"]["L_TRAJECTORIES"] = 200 # 200 equal lengths is dt = 0.02
config["DATA"]["N_TRAIN"] = 77 # Num of training traj
config["DATA"]["N_VAL"] = 11
config["DATA"]["N_TEST"] = 1
config["DATA"]["PATH"] = 'data/'
config["DATA"]["SKIP_TIME"] = 1
config["DATA"]["X1_SAMPLE_NUM"] = 13 # Num of pts per traj
config["DATA"]["X2_SAMPLE_NUM"] = 13
config["DATA"]["MAX_DELTA_T"] = 0.1 # Timestep in integrator
config["DATA"]["DT_MU"] = 0.5
config["DATA"]["DT_SIGMA"] = 0.5
config["DATA"]["REG_TIME"] = False   # True for regular sampling in time (Delta_T = const), False for random sampling in time Uniform~ (0, Delta_T)
config["DATA"]["INIT_AVAILABLE"] = True   # True for ALL initial conditions available at time 0, False for Not available
config["DATA"]["FULL_OBSERVATIONS"] = True   # Only activates for REG_TIME = False

config["PAR"] = {}
config["PAR"]["Da"] = 0.33
config["PAR"]["B"] = 11.
config["PAR"]["beta"] = 3.

config["TRAINING"] = {}
config["TRAINING"]["BATCH_SIZE"] = 256
config["TRAINING"]["LEARNING_RATE"] = 1e-2
config["TRAINING"]["EPOCHS"] = 4000 # 1000 # 1259 # 2539 # 5260

config["MODEL"] = {}
config["MODEL"]["NUM_HIDDEN"] = [64, 64]
config["MODEL"]["BOX"] = 'Black'
config["MODEL"]["Parameters"] = 'Fixed'