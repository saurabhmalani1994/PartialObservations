import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.use_deterministic_algorithms(True)

from tqdm.auto import tqdm

from utils import CSTRDataset, Network, my_Model, progress, config, get_pars, integrate_cstr


def main(config, Da=config["PAR"]["Da"]):
    
    train_loss = np.load('data/model__training_loss.npy')
    val_loss = np.load('data/model__validation_loss.npy')
    epoch_list = np.load('data/model__lr_epoch.npy')
    lr_list = np.load('data/model__lr_track.npy')
    
    f = open('minmax/maxmin.txt', 'r')
    x1max = float(f.readline())
    x1min = float(f.readline())
    x2max = float(f.readline())
    x2min = float(f.readline())
    print(x1max)
    print(x1min)
    print(x2max)
    print(x2min)
    f.close()

    minmaxes = (x1min,x1max,x2min,x2max)

    plt.figure()
    plt.semilogy(range(len(train_loss)),train_loss,label='training loss')
    plt.semilogy(range(len(val_loss)),val_loss,label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Figures/trainingloss.png')

    fig, ax2 = plt.subplots()
    ax2.semilogy(range(len(train_loss)),train_loss,label='training loss')
    ax2.semilogy(range(len(val_loss)),val_loss,label='validation loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    plt.legend()
    ax = ax2.twinx()
    ax.semilogy(epoch_list,lr_list,'g--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Learning Rate')

    plt.savefig('Figures/trainingloss_learningrate.png')
    
#     Da_list = [0.33] * 10
    Da_list = [0.2, 0.25, 0.28, 0.3, 0.32, 0.33, 0.36, 0.4, 0.42, 0.45, 0.5]#*10
#     Da_list = [0.3, 0.33, 0.36]
#     Da_list = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    
    index = 0
    
    for i in range(len(Da_list)):
        Da = Da_list[i]
        dataset_test = CSTRDataset(config["DATA"]["N_TEST"], config["DATA"]["TMAX"]*2,
                               config["DATA"]["L_TRAJECTORIES"]*2, 
                               Da=Da,
                               B=config["PAR"]["B"],
                               beta=config["PAR"]["beta"])

        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(dataset_test.tt[:-1][dataset_test.x1[0]>0], dataset_test.x1[0][dataset_test.x1[0]>0], '.-')
        ax.set_xlabel('t')
        ax.set_ylabel('X2')
        plt.savefig(config["DATA"]["PATH"]+'test_data.pdf')
        plt.show()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(dataset_test.tt[:-1], dataset_test.x2[0], '.-')
        ax.set_xlabel('t')
        ax.set_ylabel('X1')
        plt.show()
        plt.close()


        # Create the network architecture
        network = Network(hidden_cells=config["MODEL"]["NUM_HIDDEN"],
                              tau = dataset_test.delta_t,
                              B=config["PAR"]["B"],
                              beta=config["PAR"]["beta"],
                              Da=config["PAR"]["Da"],
                              minmaxes=minmaxes
                             )
        device = None
        if torch.cuda.is_available() and device is None:
            device = 'cuda'
        elif not torch.cuda.is_available() and device is None:
            device = 'cpu'
        else:
            device = device

        filename = config["DATA"]["PATH"]+'model_' +\
            '_hidden_layers_' +\
                    str(len(network.hidden_cells))+'_'+str(network.hidden_cells[0])+'.net'

        print(filename)
        network.load_state_dict(torch.load(filename, map_location=torch.device(device)))

        print(network)
        network.to(device)

        
        network.double()


        # print(len(x))

        x1_in = torch.from_numpy(dataset_test.x1[0][:]).unsqueeze(0).to(network.device)
        x2_in = torch.from_numpy(dataset_test.x2[0][:]).unsqueeze(0).to(network.device)

        x1_out, x2_out = network(x1_in,x2_in,warmup=0,dt=torch.tensor(dataset_test.delta_t).to(network.device),Da=torch.tensor(dataset_test.Da).unsqueeze(0).to(network.device))

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.plot(dataset_test.tt[1:][dataset_test.x1_out[0]>0],dataset_test.x1_out[0][dataset_test.x1_out[0]>0],'x',label='true (training points)',markersize=20,markeredgecolor='#1f77b4',markeredgewidth=2)
        ax.plot(dataset_test.tt,dataset_test.x1_data[0],'k',label='true trajectory',linewidth=3)
        ax.plot(dataset_test.tt[1:],x1_out.detach().cpu().squeeze().numpy(),'x-',label='predicted',linewidth=3)
        ax.set_xlabel(r'$t$',fontsize=40)
        ax.set_ylabel(r'$X_1$',fontsize=40)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        plt.legend(fontsize=20,loc='upper left')
        plt.title(r'$X_1$' + ' graph for Da: ' + str(dataset_test.Da),fontsize=30)
        fig.savefig('Figures/Prediction for x1 for Da_' + str(dataset_test.Da[0]) + 'index_' + str(index) + '.png',format='png')

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.plot(dataset_test.tt[1:][dataset_test.x2_out[0]>0],dataset_test.x2_out[0][dataset_test.x2_out[0]>0],'x',label='true (training points)',markersize=20,markeredgecolor='#1f77b4',markeredgewidth=2)
        ax.plot(dataset_test.tt[1:][dataset_test.x2_out[0]>0],dataset_test.x2_out[0][dataset_test.x2_out[0]>0],'k',label='true trajectory',linewidth=3)
        ax.plot(dataset_test.tt[1:],x2_out.detach().cpu().squeeze().numpy(),'x-',label='predicted',linewidth=3)
        ax.set_xlabel(r'$t$',fontsize=40)
        ax.set_ylabel(r'$X_2$',fontsize=40)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        plt.legend(fontsize=20,loc='upper left')
        plt.title(r'$X_2$' + ' graph for Da: ' + str(dataset_test.Da[0]),fontsize=30)
        fig.savefig('Figures/Prediction for x2 for Da_' + str(dataset_test.Da) + 'index_' + str(index) + '.png',format='png')
        
        index += 1
    

    
    
#     assert(False)

    x1 = []
    x2 = []

#     Da = np.concatenate((np.linspace(0.2,0.27,10,endpoint=False),
#                          np.linspace(0.27,0.29,40,endpoint=False),
#                          np.linspace(0.29,0.32,20,endpoint=False),
#                          np.linspace(0.32,0.35,20,endpoint=False),
#                          np.linspace(0.35,0.41,20,endpoint=False),
#                          np.linspace(0.41,0.43,40,endpoint=False),
#                          np.linspace(0.43,0.5,10,endpoint=True),
#                         ))
    
    Da = np.linspace(0.2,0.5,200)

    count = 0

    min_arr_x2_true = []
    max_arr_x2_true = []
    min_arr_x1_true = []
    max_arr_x1_true = []

    for i in range(len(Da)):
        dt = config["DATA"]["TMAX"] / config["DATA"]["L_TRAJECTORIES"]
        tmax = 300

        index = count

        sol = integrate_cstr(tmax=tmax, T=int(tmax/dt), Da=Da[index], B=11, beta=3)
        x1.append(sol.y[0, :])  # Observed variable at t shape (N, T)
        x2.append(sol.y[1, :])  # Observed variable at t shape (N, T)

        count += 1
        
        sol = integrate_cstr(tmax=tmax, T=int(tmax/dt), Da=Da[index], B=11, beta=3)

        min_arr_x2_true.append(np.min(sol.y[1,-int(0.05*(tmax/dt)):]))
        max_arr_x2_true.append(np.max(sol.y[1,-int(0.05*(tmax/dt)):]))
        min_arr_x1_true.append(np.min(sol.y[0,-int(0.05*(tmax/dt)):]))
        max_arr_x1_true.append(np.max(sol.y[0,-int(0.05*(tmax/dt)):]))

        if i%10 == 9:
            print(str(100*(i+1)/len(Da)) + '%')
    tt = sol.t  # Time array
    delta_t = tt[1]-tt[0]  # delta t


    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111)
    ax1.plot(Da,min_arr_x2_true,color='black')
    ax1.plot(Da,max_arr_x2_true,color='black')
    ax1.set_xlabel('Da')
    ax1.set_ylabel('X2')
    plt.show()
    
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111)
    ax1.plot(Da,min_arr_x1_true,color='black')
    ax1.plot(Da,max_arr_x1_true,color='black')
    ax1.set_xlabel('Da')
    ax1.set_ylabel('X2')
    plt.show()

    warmup_length = int(tmax/(10*dt))
    Traj_Length = int(tmax/(dt))

    percent = "{perc:.2f}%"

    max_arr_x2 = []
    min_arr_x2 = []
    
    max_arr_x1 = []
    min_arr_x1 = []
    
    

    for i in range(len(Da)):
        x1_in = torch.tensor(x1[i][0], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        x2_in = torch.tensor(x2[i][0], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        
        
        x1_out_list = []
        x2_out_list = []
        
        x1_in = x1_in.repeat(1,Traj_Length)
        x2_in = x2_in.repeat(1,Traj_Length)
        
        
        x1_in, x2_in = network(x1_in,x2_in,warmup=0,dt=torch.tensor(dataset_test.delta_t).to(network.device), 
                                   Da=torch.tensor(Da[i]).unsqueeze(0).unsqueeze(0).to(network.device))
        
#         for _ in range(Traj_Length):
# #             Da_norm = torch.tensor(Da[i], dtype=torch.float32).unsqueeze(0).unsqueeze(1).repeat((1, 1)).to(device)
            
#             x1_in, x2_in = network(x1_in,x2_in,warmup=0,dt=torch.tensor(dataset_test.delta_t).to(network.device), 
#                                    Da=torch.tensor(Da[i]).unsqueeze(0).unsqueeze(0).to(network.device))
            
            
#             x1_out_list.append(x1_in.squeeze().detach().cpu().numpy())
#             x2_out_list.append(x2_in.squeeze().detach().cpu().numpy())

        x1_out_list_arr = x1_in.squeeze().detach().cpu().numpy()
        x2_out_list_arr = x2_in.squeeze().detach().cpu().numpy()

#         x1_out_list_arr = np.array(x1_out_list)
#         x2_out_list_arr = np.array(x2_out_list)
        
        max_arr_x2.append(np.max(x2_out_list_arr[-int(tmax/(10*dt)):]))
        min_arr_x2.append(np.min(x2_out_list_arr[-int(tmax/(10*dt)):]))
        max_arr_x1.append(np.max(x1_out_list_arr[-int(tmax/(10*dt)):]))
        min_arr_x1.append(np.min(x1_out_list_arr[-int(tmax/(10*dt)):]))

        if i%10 == 9:
            print(percent.format(perc=(i+1)*100/len(Da)))
    
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111)
    ax1.plot(Da,min_arr_x2,'--',color='blue',label='Predicted')
    ax1.plot(Da,max_arr_x2,'--',color='blue')

    ax1.plot(Da,min_arr_x2_true,color='black',label='True')
    ax1.plot(Da,max_arr_x2_true,color='black')
    ax1.set_xlabel('Da',fontsize=24)
    ax1.set_ylabel(r'$X_2$',fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)
#     ax1.tick_params(axis='both', which='minor', labelsize=8)
    plt.legend(fontsize=24)
    plt.show()
    fig.savefig('Figures/Bifurcation Plot x2.png',format='png')
    
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111)
    ax1.plot(Da,min_arr_x1,'--',color='blue',label='Predicted')
    ax1.plot(Da,max_arr_x1,'--',color='blue')

    ax1.plot(Da,min_arr_x1_true,color='black',label='True')
    ax1.plot(Da,max_arr_x1_true,color='black')
    ax1.set_xlabel('Da',fontsize=24)
    ax1.set_ylabel(r'$X_1$',fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=24)
    plt.show()
    
    fig.savefig('Figures/Bifurcation Plot x1.png',format='png')


if __name__ == "__main__":
    main(config)
