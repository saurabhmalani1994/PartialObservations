import numpy as np
import matplotlib.pyplot as plt

import sys
import torch
from torch.utils.data import DataLoader
# torch.use_deterministic_algorithms(True)

# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.double)

from tqdm.auto import tqdm

from main import preprocess
from main.utils import Network, MLP, Model_Train, progress
# import preprocess
from URPModel import datagen, make_plots, helper
from config import config

# from config import config
# from utils import CSTRDataset, LSTMNetwork, LSTMModel, progress, config


def main(config, filename):
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
    
    
    # xmax = np.nanmax(np.nanmax(np.array(dataset_train.x), axis=1), axis=0)
    # xmin = np.nanmin(np.nanmin(np.array(dataset_train.x), axis=1), axis=0)
    # xmaxmin = np.savez("minmax/minmax.npz",xmax, xmin)

    f = np.load("/home/smalani/PartialObservations_URP2/minmax/minmax.npz")
    
    xmax = f['arr_0']
    xmin = f['arr_1']
    print('maxmins')
    print(xmax)
    print(xmin)
    
    norm_func = lambda input, device: (input - torch.tensor(xmin).float().to(device)) / \
                            torch.tensor((xmax - xmin)).float().to(device)
    inv_norm_func = lambda input, device: input * torch.tensor((xmax - xmin)).float().to(device) \
                                  + torch.tensor(xmin).float().to(device)
    
    
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
        else:
            raise ValueError("Tell me whether to train the parameters!")
    else:
        raise ValueError("Tell me what box to use!")
            
    
    network = my_Network(mlp, config["DATA"]["N_TRAIN"], 2, norm_func=norm_func, inv_norm_func=inv_norm_func, 
                      init_available=config["DATA"]["INIT_AVAILABLE"], integrator='RK4')
    
    print(network)
    # Move network to corresponding device (cpu or gpu)
    network.to(network.device)

    for p in network.parameters():
        print(p)
    
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
        
    model.save_network(config["DATA"]["PATH"]+filename)
        


if __name__ == "__main__":
    filename = sys.argv[1]
    main(config, filename)