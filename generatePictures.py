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
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from generator_model import Generator
from PIL import Image as im


model_path_gen_H = 'genh.pth.tar'
model_path_gen_Z = 'genz.pth.tar'

path_where_the_images_are_H = 'data/testA'
path_where_the_images_are_Z = 'data/testA'

save_fake_horses_path = 'lpips_fakes/fake_horses'
save_fake_zebras_path = 'lpips_fakes/fake_zebras'


gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

opt_gen = optim.Adam(
    list(gen_Z.parameters()) + list(gen_H.parameters()),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999),
)






load_checkpoint(
            model_path_gen_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
load_checkpoint(
            model_path_gen_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )

dataset = HorseZebraDataset(
        root_horse="data/testA", root_zebra="data/testB", transform=config.transforms
    )


loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )


for idx, (zebra, horse) in enumerate(loader):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)
        
        fake_horse = gen_H(zebra).cpu()
        fake_zebra = gen_H(horse).cpu()
        save_image(fake_horse*0.5+0.5, f"{save_fake_horses_path}/horse_{idx}.png")
        save_image(fake_zebra*0.5+0.5, f"{save_fake_zebras_path}/zebra_{idx}.png")
        
        
        
    
    
        