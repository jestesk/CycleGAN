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

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epochString):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', )

    H_reals = 0
    H_fakes = 0

    D_H_Loss_avg = 0
    D_Z_Loss_avg = 0

    cycle_Z_loss_avg = 0
    cycle_H_loss_avg = 0

    identity_Z_loss_avg = 0
    identity_H_loss_avg = 0

    G_Z_loss_avg = 0
    G_H_loss_avg = 0
    G_total_loss_avg = 0

    loop = tqdm(loader, leave=True)
    iterationCount = 0

    for idx, (zebra, horse) in enumerate(loop):
        iterationCount = idx
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()

            
            D_H_real_ground_truth = torch.zeros_like(D_H_real) if torch.rand(1).item() > 0.9 else 0.7 + torch.rand_like(D_H_real) * (1.2 - 0.7) #Flip labels 10% of the time
            D_H_fake_ground_truth = torch.ones_like(D_H_fake) if torch.rand(1).item() > 0.9 else torch.rand_like(D_H_fake) * 0.3
            D_H_real_loss = mse(D_H_real, D_H_real_ground_truth)
            D_H_fake_loss = mse(D_H_fake, D_H_fake_ground_truth)
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())

            D_Z_real_ground_truth = torch.zeros_like(D_Z_real) if torch.rand(1).item() > 0.9 else 0.7 + torch.rand_like(D_Z_real) * (1.2 - 0.7)
            D_Z_fake_ground_truth = torch.ones_like(D_Z_fake) if torch.rand(1).item() > 0.9 else torch.rand_like(D_Z_fake) * 0.3
            D_Z_real_loss = mse(D_Z_real, D_Z_real_ground_truth)
            D_Z_fake_loss = mse(D_Z_fake, D_Z_fake_ground_truth)
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_horse_loss_L1 = l1(horse, cycle_horse)
            cycle_zebra_loss_L1 = l1(zebra, cycle_zebra)
            
            cycle_horse_loss_LPIPS = lpips(horse.cpu(), cycle_horse.cpu())
            cycle_zebra_loss_LPIPS = lpips(zebra.cpu(), cycle_zebra.cpu())

            cycle_horse_loss = cycle_horse_loss_L1 + cycle_horse_loss_LPIPS
            cycle_zebra_loss = cycle_zebra_loss_L1 + cycle_zebra_loss_LPIPS

            
            

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_Z_loss = (loss_G_Z
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + identity_zebra_loss * config.LAMBDA_IDENTITY)

            G_H_loss = (
                loss_G_H
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * config.LAMBDA_IDENTITY
            )

            G_loss = G_Z_loss + G_H_loss

            D_H_Loss_avg += D_H_loss
            D_Z_Loss_avg += D_Z_loss

            cycle_Z_loss_avg += cycle_zebra_loss
            cycle_H_loss_avg += cycle_horse_loss

            identity_Z_loss_avg += identity_zebra_loss
            identity_H_loss_avg += identity_horse_loss

            G_Z_loss_avg += G_Z_loss
            G_H_loss_avg += G_H_loss
            G_total_loss_avg += G_loss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            savePath = "ProgressImages"
            if not os.path.isdir(f"{savePath}"):
                os.mkdir(f"{savePath}")

            savePath = "ProgressImages/"+epochString
            if not os.path.isdir(f"{savePath}"):
                os.mkdir(f"{savePath}")

            save_image(fake_horse*0.5+0.5, f"{savePath}/horse_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"{savePath}/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))

    D_H_Loss_avg /= iterationCount
    D_Z_Loss_avg /= iterationCount

    cycle_Z_loss_avg /= iterationCount
    cycle_H_loss_avg /= iterationCount

    identity_Z_loss_avg /= iterationCount
    identity_H_loss_avg /= iterationCount

    G_Z_loss_avg /= iterationCount
    G_H_loss_avg /= iterationCount
    G_total_loss_avg /= iterationCount

    return [D_H_Loss_avg, D_Z_Loss_avg, cycle_Z_loss_avg, cycle_H_loss_avg, identity_Z_loss_avg, identity_H_loss_avg, G_Z_loss_avg, G_H_loss_avg, G_total_loss_avg]

def StoreLosses(filename, losses):
    with open(filename, 'wb') as f:
        pickle.dump(losses, f)

def LoadLosses(filename):
    if filename in os.listdir(os.getcwd()):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return [[], [],[],[],[],[],[],[], []]
    

def main():
    startEpoch = 0
    dateString = datetime.now().strftime("%m%d%H")
    losses = LoadLosses("losses.pickle")

    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse="data/trainA", root_zebra="data/trainB", transform=config.transforms
    )
    val_dataset = HorseZebraDataset(
       root_horse="data/testA", root_zebra="data/testB", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        epochString = f"Date{dateString}Epoch{startEpoch + epoch}"

        epochLosses = train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epochString=epochString)

        for idx, loss in enumerate(losses):
            loss.append(epochLosses[idx])

        StoreLosses("losses.pickle", losses)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()
    
    a or (a and b)