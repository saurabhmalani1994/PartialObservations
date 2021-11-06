import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from config import config
from . import helper

def main(config, Da=config["PAR"]["Da"]):
    
    train_loss = np.load('data/model__training_loss.npy')
    val_loss = np.load('data/model__validation_loss.npy')
    epoch_list = np.load('data/model__lr_epoch.npy')
    lr_list = np.load('data/model__lr_track.npy', allow_pickle=True)
    
#     print(lr_list)
    
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
    
#     Da_list = [0.2] * 10
    Da_list = [0.2, 0.25, 0.28, 0.3, 0.33, 0.36, 0.4, 0.42, 0.45, 0.5]#*10
#     Da_list = [0.33]
#     Da_list = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

    helper.make_transients(Da_list)
    helper.make_RHS(Da_list)
    helper.make_Bifurc()

if __name__ == "__main__":
    main(config)
