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
        self.logtowandb = logtowandb
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
    
    def calculate_mean_std(self, dataset, device):
        print("Calculating Means and Stds of Dataset")
        means = torch.zeros(8).to(device)
        stds = torch.zeros(8).to(device)
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8 * torch.cuda.device_count())
        
        n_samples = 0
        for batch in tqdm(dataloader, desc="Calculating mean"):
            data = batch['image'].to(device)  # Shape: [batch_size, 1, 8, 2500]
            batch_size = data.size(0)
            # Reshape to [batch_size, 8, 2500]
            data = data.squeeze(1)
            # Calculate mean across batch and time dimensions
            means += data.sum(dim=(0, 2)) 
            n_samples += batch_size * data.size(2)  # batch_size * 2500
        
        # Normalize means
        means /= n_samples
        
        # Second pass: compute std
        for batch in tqdm(dataloader, desc="Calculating std"):
            data = batch['image'].to(device)  # Shape: [batch_size, 1, 8, 2500]
            data = data.squeeze(1)
            # Calculate squared differences from mean
            stds += ((data - means.view(1, 8, 1)) ** 2).sum(dim=(0, 2))
        
        # Normalize stds
        stds = torch.sqrt(stds / n_samples)
        
        print(f"Mean per lead: {means}")
        print(f"Std per lead: {stds}")
        
        return means, stds
    
    def counterfactual_sample(self, size, device):
        class_0_samples = []
        class_0_indices = []
        if self.ds is None:
            return
        for idx in range(len(self.test_ds)):
            sample = self.test_ds[idx]
            if sample['y'] == 0:  # Class 0
                class_0_samples.append(sample)
                class_0_indices.append(idx)
                if len(class_0_samples) >= size:
                    break

        # Check if we found enough class 0 samples
        if len(class_0_samples) < size:
            print(f"Warning: Only found {len(class_0_samples)} samples of class 0, fewer than batch_size={size}")

        # Create batch from collected samples
        images = torch.stack([sample['image'] for sample in class_0_samples[:size]]).to(device)
        actual_classes = torch.tensor([sample['y'] for sample in class_0_samples[:size]], dtype=torch.long).to(device)
        
        exogenous_noise, abduction_progression = self.ema.ema_model.ddim_counterfactual_sample_from_clean_to_noisy(images, sampling_ratio=self.counterfactual_sampling_ratio)
        
        init_image = exogenous_noise
        
        classes = torch.tensor([1] * size, dtype=torch.long).to(device)
        counterfactual_image, diffusion_progression = self.ema.ema_model.ddim_counterfactual_sample_from_noisy_to_counterfactual(init_image, classes, self.counterfactual_sampling_ratio, cond_scale=self.counterfactual_sampling_cond_scale)
        
        
        fig = plot_counterfactual_comparison(images, counterfactual=counterfactual_image)
        return images, exogenous_noise, counterfactual_image, fig
    
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
                        classes = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
                        classes = classes.to(self.device)
                        all_images_list = []
                        for n in batches:
                            batch_classes = classes[:n]
                            images = self.ema.ema_model.sample(batch_classes, cond_scale=6.)
                            all_images_list.append(images)
                    
                    all_images = torch.cat(all_images_list, dim = 0).cpu()
                    path = str(self.results_folder / f'sample-{milestone}.png')
                    sample_fig = plot_samples(all_images, path)
                    trainingLog['samples'] = sample_fig
                    
                    plt.close()
                    _, exogenous_noise, counterfactual_image, counterfactual_fig = self.counterfactual_sample(size=4, device=self.device)
                    trainingLog['counterfactuals'] = counterfactual_fig
                    plt.close()
                    
                    if self.logtowandb:
                        
                        wandb.log(trainingLog)

                    self.save(milestone)
                
                pbar.update(1)
        
        print(f"Training complete")