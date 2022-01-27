import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
# torch.use_deterministic_algorithms(True)

# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.double)

from tqdm.auto import tqdm

from main import preprocess
from main.utils import Network, MLP, Model_Train, progress
# import preprocess
from BandFModel import datagen#, make_plots, helper
from config import config

# from config import config
# from utils import CSTRDataset, LSTMNetwork, LSTMModel, progress, config


def main(config):
    data_train = datagen.generate_data(n_train=config["DATA"]["N_TRAIN"], duplicate_reverse=config["DATA"]["DUP_REVERSE"], detail=True)
    dataset_train = preprocess.Dataset(*data_train[:2], duplicate_reverse=config["DATA"]["DUP_REVERSE"])
    data_val = datagen.generate_data(n_train=config["DATA"]["N_VAL"],
                                    init_available=True,
                                    theta_random=True,
                                    duplicate_reverse=config["DATA"]["DUP_REVERSE"])
    dataset_val = preprocess.Dataset(*data_val, duplicate_reverse=config["DATA"]["DUP_REVERSE"])
    
    # Create PyTorch dataloaders for train and validation data
    dataloader_train = DataLoader(dataset_train, batch_size=config["TRAINING"]["BATCH_SIZE"],
                                  shuffle=True, num_workers=1, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config["TRAINING"]["BATCH_SIZE"],
                                shuffle=True, num_workers=1, pin_memory=True)
    
    
    # print('FalsityFalse')
    # print(dataset_train.x.shape)
    # np.save("data/training_data",data_train[0])
    # np.save("data/training_data_par",data_train[1])
    # np.save("data/training_data_detail_t",data_train[2])
    # np.save("data/training_data_detail_x",data_train[3])
    # np.save("data/training_data_detail_par",data_train[4])
    # assert False

    # xmax = np.nanmax(np.nanmax(np.array(dataset_train.x), axis=1), axis=0)
    # xmin = np.nanmin(np.nanmin(np.array(dataset_train.x), axis=1), axis=0)
    # xmaxmin = np.savez("minmax/minmax.npz",xmax, xmin)
    # assert False

    f = np.load("/home/smalani/PartialObservations/minmax/minmax.npz")

    xmax = f['arr_0']
    xmin = f['arr_1']
    
    print('maxmins')
    print(xmax)
    print(xmin)
    
    
    norm_func = lambda input, device: (input - torch.tensor(xmin).float().to(device)) / \
                            ((torch.tensor((xmax - xmin)).float().to(device)) + 1e-10)
    inv_norm_func = lambda input, device: input * ((torch.tensor((xmax - xmin)).float().to(device)) + 1e-10) \
                                  + torch.tensor(xmin).float().to(device)
    
    
    if config["MODEL"]["BOX"] == 'Black':
        # Create the network architecture
        mlp = MLP(6, config["MODEL"]["NUM_HIDDEN"], 6)
        
        class my_Network(Network):
            def __init__(self, network, train_size, xdim, norm_func=lambda input, device: input,
                         inv_norm_func=lambda input, device: input, init_available=True, device=None, 
                         tf_prop=1., integrator='RK4', add_par_num=0):
                super(my_Network, self).__init__(network, train_size, xdim, norm_func,
                         inv_norm_func, init_available, device, 
                         tf_prop, integrator, add_par_num)

                self.additional_pars = torch.nn.Parameter((torch.zeros(6)-1).to(self.device), requires_grad = True) 

            def output(self, x, par):

                # ANN_input = torch.cat((self.norm_func(x), par/20), dim=-1)
                ANN_input = self.norm_func(x)
                out = self.net(ANN_input)
                out = self.inv_norm_func(out) * (100 ** (self.additional_pars))

                return out
    
    elif config["MODEL"]["BOX"] == 'Grey' or config["MODEL"]["BOX"] == 'Gray':
        
        # Create the network architecture
        mlp = MLP(6, config["MODEL"]["NUM_HIDDEN"], 3)
        
        if config["MODEL"]["Parameters"] == 'Trainable':
            class my_Network(Network):
                def __init__(self, network, train_size, xdim, norm_func=lambda input, device: input,
                             inv_norm_func=lambda input, device: input, init_available=True, device=None, 
                             tf_prop=1., integrator='RK4', add_par_num=2):
                    super(my_Network, self).__init__(network, train_size, xdim, norm_func,
                             inv_norm_func, init_available, device, 
                             tf_prop, integrator, add_par_num)

                    self.additional_pars = torch.nn.Parameter((torch.cat(((torch.tensor([0.4476, -0.4859, -0.6419])), 
                                                                          (torch.zeros(4) + 1)))).to(self.device), 
                                                                requires_grad = True)

                # def output(self, x_input, par):

                #     ANN_input = self.norm_func(x_input)[...,[0,2,3,4,5]]
                #     ANN_output = self.net(ANN_input) * (2 ** (7 * self.additional_pars[...,:3]))

                #     x, y, z, u, v, g = torch.unbind(x_input, dim=-1)
                #     u1_prime, u2_prime, u3_prime = torch.unbind(ANN_output, dim=-1)
                    
                #     omega = self.additional_pars[-4] * 10
                #     sigma = self.additional_pars[-3] * 10
                #     rho = self.additional_pars[-2]
                #     eta = self.additional_pars[-1] * 10

                #     alpha, uf, _, _, _, _, _, _, uc1_prime, uc2_prime, uc3_prime = datagen.par_fun()

                #     output = []

                #     output.append(-alpha * x + u1_prime * x - uc1_prime * x)
                #     output.append(-alpha * y + u2_prime * y - uc2_prime * y)
                #     output.append(-alpha * z + u3_prime * z - uc3_prime * z)
                #     output.append(alpha * (uf - u) - u1_prime * x)
                #     output.append(-alpha * v + omega * u1_prime * x - u2_prime * y - sigma * u3_prime * z)
                #     output.append(-alpha * g + rho * u2_prime * y + eta * u3_prime * z)


                #     out = torch.stack((output), dim=-1)
                #     return out

                def output(self, x_input, par):
                    ANN_input = self.norm_func(x_input)
                    ANN_output = self.net(ANN_input)
                    ANN_output = ANN_output * (100 ** (self.additional_pars[:3]))

                    x, y, z, u, v, g = torch.unbind(x_input, dim=-1)

                    u1_prime_x, u2_prime_y, u3_prime_z = torch.unbind(ANN_output, dim=-1)                

                    omega = self.additional_pars[-4] * 10
                    sigma = self.additional_pars[-3] * 10
                    rho = self.additional_pars[-2] / 10
                    eta = self.additional_pars[-1]

                    alpha, uf, _, _, _, _, _, _, uc1_prime, uc2_prime, uc3_prime = datagen.par_fun()

                    output = []

                    output.append(-alpha * x + u1_prime_x - uc1_prime * x)
                    output.append(-alpha * y + u2_prime_y - uc2_prime * y)
                    output.append(-alpha * z + u3_prime_z - uc3_prime * z)
                    output.append(alpha * (uf - u) - u1_prime_x)
                    output.append(-alpha * v + omega * u1_prime_x - u2_prime_y - sigma * u3_prime_z)
                    output.append(-alpha * g + rho * u2_prime_y + eta * u3_prime_z)

                    out = torch.stack((output), dim=-1)
                    return out
                def raw_output(self, x_input, par):

                    ANN_input = self.norm_func(x_input)
                    ANN_output = self.net(ANN_input) * (2 ** (7 * self.additional_pars))

                    return ANN_output
        elif config["MODEL"]["Parameters"] == 'Fixed':
            class my_Network(Network):
                def __init__(self, network, train_size, xdim, norm_func=lambda input, device: input,
                             inv_norm_func=lambda input, device: input, init_available=True, device=None, 
                             tf_prop=1., integrator='RK4', add_par_num=2):
                    super(my_Network, self).__init__(network, train_size, xdim, norm_func,
                             inv_norm_func, init_available, device, 
                             tf_prop, integrator, add_par_num)

                    # self.additional_pars = torch.nn.Parameter((torch.zeros(2)-0.5).to(self.device), 
                    #                             requires_grad = True)
                    self.additional_pars = torch.nn.Parameter((torch.tensor([0.4476, -0.4859, -0.6419])).to(self.device), 
                                                                    requires_grad = True)

                    

                def output(self, x_input, par):
                    
                    # print('outputs')
                    ANN_input = self.norm_func(x_input)
                    ANN_output = self.net(ANN_input)
                    # print(ANN_output)
                    ANN_output = ANN_output * (100 ** (self.additional_pars))
                    # print(ANN_output)
                    # assert False

                    x, y, z, u, v, g = torch.unbind(x_input, dim=-1)

                    # u1_prime, u2_prime, u3_prime = torch.unbind(ANN_output, dim=-1)
                    u1_prime_x, u2_prime_y, u3_prime_z = torch.unbind(ANN_output, dim=-1)

                    # print('The us')
                    # print(u1_prime_x)
                    # print(u3_prime_z)

                    # u2_prime = torch.zeros(u1_prime.shape).to(self.device)
                    

                    alpha, uf, omega, sigma, rho, eta, phi1, phi2, uc1_prime, uc2_prime, uc3_prime = datagen.par_fun(D=1/7.3, sf=2.5)

                    # u2_prime_y = y * phi1 * v / (1+v)

                    output = []

                    # output.append(-alpha * x + u1_prime * x - uc1_prime * x)
                    # output.append(torch.zeros(output[-1].shape).to(self.device))
                    # output.append(-alpha * z + u3_prime * z - uc3_prime * z)
                    # output.append(alpha * (uf - u) - u1_prime * x)
                    # output.append(-alpha * v + omega * u1_prime * x - u2_prime * y - sigma * u3_prime * z)
                    # output.append(-alpha * g + rho * u2_prime * y + eta * u3_prime * z)

                    output.append(-alpha * x + u1_prime_x - uc1_prime * x)

                    output.append(-alpha * y + u2_prime_y - uc2_prime * y)
                    # output.append(torch.zeros(output[-1].shape).to(self.device))

                    output.append(-alpha * z + u3_prime_z - uc3_prime * z)
                    output.append(alpha * (uf - u) - u1_prime_x)
                    output.append(-alpha * v + omega * u1_prime_x - u2_prime_y - sigma * u3_prime_z)
                    output.append(-alpha * g + rho * u2_prime_y + eta * u3_prime_z)

                    out = torch.stack((output), dim=-1)
                    return out
                def raw_output(self, x_input, par):

                    ANN_input = self.norm_func(x_input)
                    ANN_output = self.net(ANN_input) * (2 ** (7 * self.additional_pars))

                    return ANN_output

        else:
            raise ValueError("Tell me whether to train the parameters!")
    else:
        raise ValueError("Tell me what box to use!")
            
    
    if config["DATA"]["DUP_REVERSE"]:
        network = my_Network(mlp, config["DATA"]["N_TRAIN"]*2, 6, norm_func=norm_func, inv_norm_func=inv_norm_func, 
                        init_available=config["DATA"]["INIT_AVAILABLE"], integrator='RK4')
    else:
        network = my_Network(mlp, config["DATA"]["N_TRAIN"], 6, norm_func=norm_func, inv_norm_func=inv_norm_func, 
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
        val_loss = model.validate(epoch)
        train_loss = model.train(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        progress_bar.set_description(progress(train_loss, val_loss))
#         if epoch == 20:
#             assert False
        
    model.save_network('data/'+'model_')
        



if __name__ == "__main__":
    main(config)
