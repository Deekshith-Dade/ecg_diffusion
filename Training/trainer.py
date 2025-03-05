import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from ema_pytorch import EMA

from utils.plot_utils import plot_samples

from torchvision import transforms as T, utils
import math
from pathlib import Path
from tqdm.auto import tqdm
import itertools

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
        dataset = None
    ):
        super().__init__()
        
        assert dataset is not None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ids = range(torch.cuda.device_count())
        
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        
        self.batch_size = train_batch_size
        
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        
        self.ds = dataset
        
        mean, std = self.calculate_mean_std(dataset, self.device)
        diffusion_model._set_mean_std(mean, std)
        
        print(f"Number of Training Examples: {len(self.ds)}")
        self.dataloader = DataLoader(self.ds, batch_size=train_batch_size, shuffle = True, pin_memory = True, num_workers=16)
        
        self.dataloader_iter  = itertools.cycle(self.dataloader)

        if torch.cuda.is_available():
            diffusion_model.to(self.device)
            self.model = torch.nn.DataParallel(diffusion_model, self.device_ids)
        else:
            self.model = diffusion_model
        
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every = ema_update_every)
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        
        self.step = 0
    
    def calculate_mean_std(self, dataset, device):
        print("Calculating Means and Stds of Dataset")
        means = torch.zeros(8).to(device)
        stds = torch.zeros(8).to(device)
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        
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
    
    def save(self, milestone):
        
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'beta_schedule': self.model.module.beta_schedule,
        }
        
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    
    def train(self):
        losses = []
        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:
            
            while self.step < self.train_num_steps:
                total_loss = 0.
                
                data = next(self.dataloader_iter)
                
                images = data['image']
                classes = data['y']
                
                images = images.to(self.device)
                classes = classes.to(self.device)
                
                loss = self.model(images, classes=classes)
                losses.append(loss.item())
                total_loss += loss.item()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                
                loss.backward()
                
                pbar.set_description(f'loss: {total_loss:.6f}')
                
                self.opt.step()
                self.opt.zero_grad()
                
                self.step += 1

                # Saving samples etc
                self.ema.to(self.device)
                self.ema.update()
                
                if self.step != 0 and self.step % self.save_and_sample_every == 0:
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
                    plot_samples(all_images, path)
                    # utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                    self.save(milestone)
                
                pbar.update(1)
        
        print(f"Training complete")