import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
# torch.use_deterministic_algorithms(True)

# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.double)

from tqdm.auto import tqdm



from utils import CSTRDataset, Network, my_Model, progress, config
# from utils import CSTRDataset, LSTMNetwork, LSTMModel, progress, config


def main(config):
    # Create CSTR Datasets, which are required for the data loader.
    dataset_train = CSTRDataset(config["DATA"]["N_TRAIN"], config["DATA"]["TMAX"],
                                config["DATA"]["L_TRAJECTORIES"], 
                           Da=config["PAR"]["Da"],
                           B=config["PAR"]["B"],
                           beta=config["PAR"]["beta"])
    dataset_val = CSTRDataset(config["DATA"]["N_VAL"], config["DATA"]["TMAX"],
                              config["DATA"]["L_TRAJECTORIES"], 
                           Da=config["PAR"]["Da"],
                           B=config["PAR"]["B"],
                           beta=config["PAR"]["beta"], random=True)
    dataset_test = CSTRDataset(config["DATA"]["N_TEST"], config["DATA"]["TMAX"],
                           config["DATA"]["L_TRAJECTORIES"], 
                           Da=config["PAR"]["Da"],
                           B=config["PAR"]["B"],
                           beta=config["PAR"]["beta"])
    
    # Create PyTorch dataloaders for train and validation data
    dataloader_train = DataLoader(dataset_train, batch_size=config["TRAINING"]["BATCH_SIZE"],
                                  shuffle=True, num_workers=1, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config["TRAINING"]["BATCH_SIZE"],
                                shuffle=False, num_workers=1, pin_memory=True)

    x1max = np.max(dataset_train.x1)
    x1min = np.min(dataset_train.x1)

    x2max = np.max(dataset_train.x2)
    x2min = np.min(dataset_train.x2)
    
    f = open('minmax/maxmin.txt','w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
    f.write('{}\n'.format(x1max))
    f.write('{}\n'.format(x1min))
    f.write('{}\n'.format(x2max))
    f.write('{}\n'.format(x2min))
    f.close()
    
    minmaxes = (x1min, x1max, x2min, x2max)
    
#     plt.figure()
#     plt.plot(dataset_train.tt[:-1][dataset_train.x1[0]>0],dataset_train.x1[0][dataset_train.x1[0]>0],'x-')
#     plt.xlabel('Time')
#     plt.ylabel('x1')
    
#     plt.figure()
#     plt.plot(dataset_train.tt[:-1],dataset_train.x2[0],'x-')
#     plt.xlabel('Time')
#     plt.ylabel('x2')
    
#     assert(False)
    
    # Create the network architecture
    network = Network(hidden_cells=config["MODEL"]["NUM_HIDDEN"],
                      tau = dataset_train.delta_t,
                      B=config["PAR"]["B"],
                      beta=config["PAR"]["beta"],
                      Da=config["PAR"]["Da"],
                      minmaxes=minmaxes
                     )
    print(network)
    # Move network to corresponding device (cpu or gpu)
    network.to(network.device)
    
#     print(network.layers)

    # Create model wrapper around architecture
    # Contains train and validation functions
    model = my_Model(dataloader_train, dataloader_val, network,
                      learning_rate=config["TRAINING"]["LEARNING_RATE"])

    # Train for the given number of epochs
    progress_bar = tqdm(range(0, config["TRAINING"]["EPOCHS"]),
                        leave=True, position=0, desc=progress(0, 0))
    train_loss_list = []
    val_loss_list = []
    for epoch in progress_bar:
        train_loss = model.train(epoch)
        val_loss = model.validate(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        progress_bar.set_description(progress(train_loss, val_loss))
    model.save_network(config["DATA"]["PATH"]+'model_')



if __name__ == "__main__":
    main(config)
