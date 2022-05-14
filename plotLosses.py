import pickle
import os
import time
import matplotlib.pyplot as plt
import torch

available_losses = ["D_H_Loss_avg", "D_Z_Loss_avg", "cycle_Z_loss_avg", "cycle_H_loss_avg", "identity_Z_loss_avg", "identity_H_loss_avg", "G_Z_loss_avg", "G_H_loss_avg", "G_total_loss_avg"]

chosen_losses = ["D_Z_Loss_avg", "cycle_Z_loss_avg", "cycle_H_loss_avg", "identity_Z_loss_avg"]

lossesDict = {'LPIPS+L1 Cycle Loss' : 'Losses\losses_L1_and_lpips_cycleLoss.pickle',
              'TestComp' : 'Losses\losses_L1_and_lpips_cycleLoss.pickle'
              }

def load_loss_from_pickle(path):
    with open(path, 'rb') as f:
            D_H_Loss_avg, D_Z_Loss_avg, cycle_Z_loss_avg, cycle_H_loss_avg, identity_Z_loss_avg, identity_H_loss_avg, G_Z_loss_avg, G_H_loss_avg, G_total_loss_avg = pickle.load(f)
    return { 
        "D_H_Loss_avg": D_H_Loss_avg,
        "D_Z_Loss_avg": D_Z_Loss_avg,
        "cycle_Z_loss_avg":cycle_Z_loss_avg,
        "cycle_H_loss_avg":cycle_H_loss_avg,
        "identity_Z_loss_avg":identity_Z_loss_avg,
        "identity_H_loss_avg":identity_H_loss_avg,
        "G_Z_loss_avg": G_Z_loss_avg,
        "G_H_loss_avg":G_H_loss_avg,
        "G_total_loss_avg":G_total_loss_avg,
    }



def plot_losses(lossesDict, chosen_losses):
    for loss in chosen_losses:
        plt.figure() 
        for key in lossesDict.keys():
            a = lossesDict[key][loss]
            data =  [x.cpu().detach().numpy() for x in a]
            plt.plot(data, label = key)
        
        
        plt.xlabel('epochs')
        plt.ylabel(loss)
        plt.legend(loc='upper center')
        plt.ion()
        plt.show()
        plt.show(block=False)
        plt.pause(0.0001)
    
   
    

def Main(losseDict, chosen_losses):
    for key in losseDict.keys():    
        lossesDict[key] = load_loss_from_pickle(lossesDict[key])
        
    plot_losses(lossesDict, chosen_losses)
  
    
Main(lossesDict, chosen_losses)
hello = 3


    
    
    

    