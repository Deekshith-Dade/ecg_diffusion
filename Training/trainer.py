from numpy import std
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from zmq import device
import counterfactual
from ema_pytorch import EMA

from utils.plot_utils import plot_samples

from torchvision import transforms as T, utils
import math
from pathlib import Path
from tqdm.auto import tqdm
import itertools
import wandb

import matplotlib.pyplot as plt

from utils.plot_utils import plot_counterfactual_comparison

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        # folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr = 1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        dataset = None,
        test_dataset = None,
        sampling_cond_scale = 6.,
        counterfactual_sampling_ratio = 0.75,
        counterfactual_sampling_cond_scale = 6.,
        logtowandb = False
    ):
        super().__init__()
        
        assert dataset is not None and test_dataset is not None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ids = range(torch.cuda.device_count())
        
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        
        self.batch_size = train_batch_size
        
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        
        self.ds = dataset
        self.test_ds = test_dataset
        
        self.mean, self.std = self.calculate_mean_std(dataset, self.device)
        diffusion_model._set_mean_std(self.mean, self.std)
        self.logtowandb = logtowandb
        if self.logtowandb:
            wandb.config.update({
                "lead_means": self.mean.cpu().numpy(),
                "lead_stds": self.std.cpu().numpy()
            })
        
        print(f"Number of Training Examples: {len(self.ds)}")
        self.dataloader = DataLoader(self.ds, batch_size=train_batch_size, shuffle = True, pin_memory = True, num_workers=16 * torch.cuda.device_count())
        
        self.dataloader_iter  = itertools.cycle(self.dataloader)

        if torch.cuda.is_available():
            diffusion_model.to(self.device)
            self.model = torch.nn.DataParallel(diffusion_model, self.device_ids)
        else:
            self.model = diffusion_model
        
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every = ema_update_every)
        
        self.sampling_cond_scale = sampling_cond_scale
        self.counterfactual_sampling_ratio = counterfactual_sampling_ratio
        self.counterfactual_sampling_cond_scale = counterfactual_sampling_cond_scale
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
    
    def calculate_mean_std(self, dataset, device):
        print("Calculating Means and Stds of Dataset")
        
        means = torch.zeros(8, device=device)
        stds = torch.zeros(8, device=device)
        n_samples = 0

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8 * torch.cuda.device_count())

        for batch in tqdm(dataloader, desc="Calculating mean & std"):
            data = batch['image'].to(device)  # Shape: [batch_size, 1, 8, 2500]
            batch_size = data.size(0)
            data = data.squeeze(1)  # Shape: [batch_size, 8, 2500]
            
            means += data.sum(dim=(0, 2))  # Sum over batch and time dimensions
            stds += (data ** 2).sum(dim=(0, 2))  # Sum of squares
            n_samples += batch_size * data.size(2)  # batch_size * 2500
        
        means /= n_samples  # Final mean
        stds = torch.sqrt((stds / n_samples) - means ** 2)  # Variance formula

        print(f"Mean per lead: {means}")
        print(f"Std per lead: {stds}")
        
        return means, stds

    
    def counterfactual_sample(self, size, device, trainingLog=None):
        class_0_samples = []
        class_0_indices = []
        class_1_samples = []
        if self.ds is None:
            return
        for idx in range(len(self.test_ds)):
            sample = self.test_ds[idx]
            if sample['y'] == 0 and len(class_0_samples) < size:  # Class 0
                class_0_samples.append(sample)
            if sample['y'] == 1 and len(class_1_samples) < size:
                class_1_samples.append(sample)
            if len(class_1_samples) >= size and len(class_0_samples) >= size:
                break

        # Check if we found enough class 0 samples
        if len(class_0_samples) < size or len(class_1_samples) < size:
            print(f"Warning: Less Samples found than {size}")

        # Create batch from collected samples
        unhealthy_images = torch.stack([sample['image'] for sample in class_0_samples[:size]]).to(device)
        unhealthy_kcls = torch.stack([sample['kclVal'] for sample in class_0_samples[:size]]).to(device)
        
        healthy_images = torch.stack([sample['image'] for sample in class_1_samples[:size]]).to(device)
        healthy_kcls = torch.stack([sample['kclVal'] for sample in class_1_samples[:size]]).to(device)
        
        # Unhealthy to Healthy Counterfactual
        exogenous_noise, abduction_progression = self.ema.ema_model.ddim_counterfactual_sample_from_clean_to_noisy(unhealthy_images, sampling_ratio=self.counterfactual_sampling_ratio)
        init_image = exogenous_noise
        classes = torch.tensor([1] * size, dtype=torch.long).to(device)
        counterfactual_image, diffusion_progression = self.ema.ema_model.ddim_counterfactual_sample_from_noisy_to_counterfactual(init_image, classes, self.counterfactual_sampling_ratio, cond_scale=self.counterfactual_sampling_cond_scale)
       
        un_kv = dict(
            key = 'KCLs',
            values = unhealthy_kcls
        )
        unhealth_to_healthy_fig = plot_counterfactual_comparison(unhealthy_images, counterfactual=counterfactual_image, kv=un_kv)
        if trainingLog:
            trainingLog['Unhealthy To Healthy'] = unhealth_to_healthy_fig
            plt.close()
        
        # Healthy to Unhealthy Counterfactual
        exogenous_noise, abduction_progression = self.ema.ema_model.ddim_counterfactual_sample_from_clean_to_noisy(healthy_images, sampling_ratio=self.counterfactual_sampling_ratio)
        init_image = exogenous_noise
        classes = torch.tensor([0] * size, dtype=torch.long).to(device)
        counterfactual_image, diffusion_progression = self.ema.ema_model.ddim_counterfactual_sample_from_noisy_to_counterfactual(init_image, classes, self.counterfactual_sampling_ratio, cond_scale=self.counterfactual_sampling_cond_scale)
       
        he_kv = dict(
            key = 'KCLs',
            values = healthy_kcls
        )
        healthy_to_unhealthy_fig = plot_counterfactual_comparison(unhealthy_images, counterfactual=counterfactual_image, kv=he_kv)
        if trainingLog:
            trainingLog['Healthy to Unhealthy'] = healthy_to_unhealthy_fig
            plt.close()
        return unhealth_to_healthy_fig, healthy_to_unhealthy_fig
    
    def save(self, milestone):
        
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'mean': self.mean,
            'std': self.std,
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'beta_schedule': self.model.module.beta_schedule,
        }
        
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    
    def train(self):
        losses = []
        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:
            
            while self.step < self.train_num_steps:
                
                data = next(self.dataloader_iter)
                
                images = data['image']
                classes = data['y']
                
                images = images.to(self.device)
                classes = classes.to(self.device)
                
                loss = self.model(images, classes=classes)
                loss = loss.mean()
                
                losses.append(loss.item())
                
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                loss.backward()
                
                pbar.set_description(f'loss: {loss.item():.6f}')
                
                self.opt.step()
                self.opt.zero_grad()
                
                self.step += 1

                # Saving samples etc
                self.ema.to(self.device)
                self.ema.update()
                
                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    trainingLog = dict(
                        step = self.step,
                        train_loss = torch.tensor(losses).mean(),
                    )
                    losses = []
                    self.ema.ema_model.eval()
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 5))
                    plt.plot(losses[900:])
                    plt.title('Training Loss Over Time')
                    plt.xlabel('Step')
                    plt.ylabel('Loss')
                    plt.savefig(str(self.results_folder / f'loss.png'))
                    plt.close()
                    with torch.no_grad():
                        milestone = self.step // self.save_and_sample_every
                        batches = num_to_groups(self.num_samples, self.batch_size)
                        classes = torch.tensor([1, 0, 1, 0, 1], dtype=torch.long)
                        classes = classes.to(self.device)
                        all_images_list = []
                        for n in batches:
                            batch_classes = classes[:n]
                            images = self.ema.ema_model.sample(batch_classes, cond_scale=6.)
                            all_images_list.append(images)
                    
                    all_images = torch.cat(all_images_list, dim = 0).cpu()
                    kv = dict(
                        key = 'class',
                        values = classes
                    )
                    path = str(self.results_folder / f'sample-{milestone}.png')
                    sample_fig = plot_samples(all_images, path, kv=kv)
                    trainingLog['samples'] = sample_fig
                    
                    plt.close()
                    _ = self.counterfactual_sample(size=4, device=self.device, trainingLog=trainingLog)
                    # trainingLog['counterfactuals'] = counterfactual_fig
                    # plt.close()
                    
                    if self.logtowandb:
                        
                        wandb.log(trainingLog)

                    self.save(milestone)
                
                pbar.update(1)
        
        print(f"Training complete")