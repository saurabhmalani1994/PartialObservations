import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
# torch.use_deterministic_algorithms(True)

# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.double)

from tqdm.auto import tqdm


from utils import Network, MLP, Model_Train, progress
import preprocess
import datagen
from config import config
# from utils import CSTRDataset, LSTMNetwork, LSTMModel, progress, config


def main(config):
    data_train = datagen.generate_data(n_train=config["DATA"]["N_TRAIN"])
    dataset_train = preprocess.Dataset(*data_train)
    data_val = datagen.generate_data(n_train=config["DATA"]["N_VAL"],
                                    init_available=True,
                                    Da_random=True)
    dataset_val = preprocess.Dataset(*data_val)
    
    # Create PyTorch dataloaders for train and validation data
    dataloader_train = DataLoader(dataset_train, batch_size=config["TRAINING"]["BATCH_SIZE"],
                                  shuffle=True, num_workers=1, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config["TRAINING"]["BATCH_SIZE"],
                                shuffle=True, num_workers=1, pin_memory=True)
    
    
    xmax = np.nanmax(np.nanmax(np.array(dataset_train.x), axis=1), axis=0)
    xmin = np.nanmin(np.nanmin(np.array(dataset_train.x), axis=1), axis=0)
    xmaxmin = np.savez("minmax/minmax.npz",xmax, xmin)
    
    print('maxmins')
    print(xmax)
    print(xmin)
    
    norm_func = lambda input, device: (input - torch.tensor(xmin).float().to(device)) / \
                            torch.tensor((xmax - xmin)).float().to(device)
    inv_norm_func = lambda input, device: input * torch.tensor((xmax - xmin)).float().to(device) \
                                  + torch.tensor(xmin).float().to(device)
    
    # Create the network architecture
    mlp = MLP(3, [64, 64], 2)
    network = Network(mlp, config["DATA"]["N_TRAIN"], 2, norm_func=norm_func, inv_norm_func=inv_norm_func, 
                      init_available=config["DATA"]["INIT_AVAILABLE"], integrator='RK4')
    
    print(network)
    # Move network to corresponding device (cpu or gpu)
    network.to(network.device)
    
#     print(network.layers)

    # Create model wrapper around architecture
    # Contains train and validation functions
    model = Model_Train(dataloader_train, dataloader_val, network,
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
#         if epoch == 20:
#             assert False
        
    model.save_network(config["DATA"]["PATH"]+'model_')
        



if __name__ == "__main__":
    main(config)
