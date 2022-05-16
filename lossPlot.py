import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import os
from datetime import datetime
import pickle
from matplotlib import pyplot as plt

def LoadLosses(filename):
    if filename in os.listdir(os.getcwd()):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return [[], [],[],[],[],[],[],[], []]

def PlotLoss(index):
    lossList = losses[index]
    lossList = list(map(lambda e: e.item(), lossList))
    plt.plot(list(range(len(lossList) - (startEpochIndex))), lossList[startEpochIndex:])   
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

lossTitles = ["D_H_Loss_avg", "D_Z_Loss_avg", "cycle_Z_loss_avg", "cycle_H_loss_avg", "identity_Z_loss_avg", "identity_H_loss_avg", "G_Z_loss_avg", "G_H_loss_avg", "G_total_loss_avg"]
losses = LoadLosses("losses.pickle")
startEpochIndex = 0

# Total loss
PlotLoss(-1)
plt.title(lossTitles[-1])
plt.savefig("Total loss.png")

# Horse D vs G loss
plt.figure()
PlotLoss(0)
PlotLoss(-2)
plt.title("Horse Discriminator Vs Generator")
plt.legend(["Discriminator", "Generator"])
plt.savefig("Horse Discriminator Vs Generator.png")

# Zebra D vs G loss
plt.figure()
PlotLoss(1)
PlotLoss(-3)
plt.title("Zebra Discriminator Vs Generator")
plt.legend(["Discriminator", "Generator"])
plt.savefig("Zebra Discriminator Vs Generator.png")

# Horse Identity
plt.figure()
PlotLoss(-4)
plt.title("Horse Identity Loss")
plt.savefig("Horse Identity Loss.png")

# Zebra Identity
plt.figure()
PlotLoss(-5)
plt.title("Zebra Identity Loss")
plt.savefig("Zebra Identity Loss.png")

# Horse Cycle
plt.figure()
PlotLoss(3)
plt.title("Horse Cycle Loss")
plt.savefig("Horse Cycle Loss.png")

# Zebra Cycle
plt.figure()
PlotLoss(2)
plt.title("Zebra Cycle Loss")
plt.savefig("Zebra Cycle Loss.png")