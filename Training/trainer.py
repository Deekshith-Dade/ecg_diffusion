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
from sklearn.metrics import roc_auc_score


import matplotlib.pyplot as plt

from utils.plot_utils import plot_counterfactual_comparison
from classification.classify import create_model

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
        logtowandb = False,
        eval_classification_model_path = None
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
        
        self.classification_model_path = eval_classification_model_path
        
        self.step = 0

    def assign_classification_model(self, model_path):
        model = create_model(model_path, baseline=False, finetune=False)
        model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        model.to(self.device)
        self.classification_model = model

    def gather_samples(self, size):
        class_0_samples = []
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
        
        return class_0_samples, class_1_samples
    
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
        class_0_samples, class_1_samples = self.gather_samples(size)

        # Create batch from collected samples
        unhealthy_images = torch.stack([sample['image'] for sample in class_0_samples[:size]]).to(device)
        unhealthy_vals = torch.stack([sample['val'] for sample in class_0_samples[:size]]).to(device)
        
        healthy_images = torch.stack([sample['image'] for sample in class_1_samples[:size]]).to(device)
        healthy_vals = torch.stack([sample['val'] for sample in class_1_samples[:size]]).to(device)
        
        # Unhealthy to Healthy Counterfactual
        exogenous_noise, abduction_progression = self.ema.ema_model.ddim_counterfactual_sample_from_clean_to_noisy(unhealthy_images, sampling_ratio=self.counterfactual_sampling_ratio)
        init_image = exogenous_noise
        classes = torch.tensor([1] * size, dtype=torch.long).to(device)
        counterfactual_image, diffusion_progression = self.ema.ema_model.ddim_counterfactual_sample_from_noisy_to_counterfactual(init_image, classes, self.counterfactual_sampling_ratio, cond_scale=self.counterfactual_sampling_cond_scale)
       
        un_kv = dict(
            key = class_0_samples[0]['key'],
            values = unhealthy_vals
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
            key = class_1_samples[0]['key'],
            values = healthy_vals
        )
        healthy_to_unhealthy_fig = plot_counterfactual_comparison(unhealthy_images, counterfactual=counterfactual_image, kv=he_kv)
        if trainingLog:
            trainingLog['Healthy to Unhealthy'] = healthy_to_unhealthy_fig
            plt.close()
        return unhealth_to_healthy_fig, healthy_to_unhealthy_fig


    def evaluation_classification_model(self, trainingLog=None):
        print("Evaluating Classification Model")
        if self.classification_model_path is None:
            print("No classification model path provided. Skipping evaluation.")
            return
        
        if not hasattr(self, 'classification_model'):
            self.assign_classification_model(self.classification_model_path)
        
        
        class_0_samples, class_1_samples = self.gather_samples(100)
        # Create batch from collected samples
        unhealthy_images = torch.stack([sample['image'] for sample in class_0_samples[:100]]).to(self.device)
        unhealthy_vals = torch.tensor([sample['y'] for sample in class_0_samples[:100]], dtype=torch.long)
        healthy_images = torch.stack([sample['image'] for sample in class_1_samples[:100]]).to(self.device)
        healthy_vals = torch.tensor([sample['y'] for sample in class_1_samples[:100]], dtype=torch.long)
        
        # Unhealthy to Healthy Counterfactual
        exogenous_noise, abduction_progression = self.ema.ema_model.ddim_counterfactual_sample_from_clean_to_noisy(unhealthy_images, sampling_ratio=self.counterfactual_sampling_ratio)
        init_image = exogenous_noise
        classes = torch.tensor([1] * 100, dtype=torch.long).to(self.device)
        h_counterfactual_image, diffusion_progression = self.ema.ema_model.ddim_counterfactual_sample_from_noisy_to_counterfactual(init_image, classes, self.counterfactual_sampling_ratio, cond_scale=self.counterfactual_sampling_cond_scale)
        
        unhealthy_images = unhealthy_images.cpu()
        h_counterfactual_image = h_counterfactual_image.cpu()
        
        
        # Healthy to Unhealthy Counterfactual
        exogenous_noise, abduction_progression = self.ema.ema_model.ddim_counterfactual_sample_from_clean_to_noisy(healthy_images, sampling_ratio=self.counterfactual_sampling_ratio)
        init_image = exogenous_noise
        classes = torch.tensor([0] * 100, dtype=torch.long).to(self.device)
        unh_counterfactual_image, diffusion_progression = self.ema.ema_model.ddim_counterfactual_sample_from_noisy_to_counterfactual(init_image, classes, self.counterfactual_sampling_ratio, cond_scale=self.counterfactual_sampling_cond_scale)
        
        healthy_images = healthy_images.cpu()
        unh_counterfactual_image = unh_counterfactual_image.cpu()
        
        # Concatenate all batches
        X_orig = torch.cat([unhealthy_images, healthy_images], dim=0)
        y_orig = torch.cat([unhealthy_vals, healthy_vals], dim=0)
        X_counterfactual = torch.cat([h_counterfactual_image, unh_counterfactual_image], dim=0)
        
        self.evaluate_counterfactuals(X_orig, X_counterfactual, y_orig, trainingLog=trainingLog)
        
        torch.cuda.empty_cache()
        
        
        
    def plot_comparisons(self, probs_orig, probs_cf, y_orig, trainingLog=None):
        import matplotlib.pyplot as plt
        import numpy as np

        # Convert to NumPy
        probs_orig_np = probs_orig.cpu().numpy()
        probs_cf_np = probs_cf.cpu().numpy()
        y_orig_np = y_orig.cpu().numpy()

        # Compute delta
        delta = probs_cf_np - probs_orig_np

        # Create color map: 1 = normal (green), 0 = high (red)
        colors = np.where(y_orig_np == 1, 'green', 'red')
        labels = np.where(y_orig_np == 1, 'Normal', 'High')

        # Create figure and axes objects
        fig, ax = plt.subplots(figsize=(12, 5))
        
        for i in range(len(delta)):
            ax.plot(i, delta[i], marker='o', color=colors[i], alpha=0.7)

        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title("Change in P(normal) per sample (CF - Original)")
        ax.set_ylabel("ΔP(normal)")
        ax.set_xlabel("Sample Index")
        # legend_handles = [
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Original: Normal'),
        #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Original: High')
        # ]
        # ax.legend(handles=legend_handles)
        ax.grid(True)
        fig.tight_layout()
        
        if trainingLog is not None:
            trainingLog['comparison_plot'] = fig
        
        fig.savefig(str(self.results_folder / f'comparison_plot.png'))
        plt.close(fig)


    def evaluate_counterfactuals(self, X_orig, X_cf, y_orig, trainingLog=None):
        # Evaluate the classification model on original and counterfactual samples
        X_orig = X_orig.squeeze(1)
        X_cf = X_cf.squeeze(1)
        self.classification_model.eval()
        with torch.no_grad():
           X_orig = X_orig.to(self.device)
           probs_orig = self.classification_model(X_orig)
           del X_orig
           X_cf = X_cf.to(self.device)
           
           probs_cf = self.classification_model(X_cf)
           del X_cf
           
           self.plot_comparisons(probs_orig, probs_cf, y_orig, trainingLog=trainingLog)
           
           y_orig = y_orig.int()
           
           auc_orig = roc_auc_score(y_orig.cpu().numpy(), probs_orig.cpu().numpy())
           auc_cf = roc_auc_score(1 - y_orig.cpu().numpy(), probs_cf.cpu().numpy())
           
           trainingLog['AUC (Original)'] = auc_orig
           trainingLog['AUC (Counterfactual)'] = auc_cf
           
           print(f"AUC (Original): {auc_orig}")
           print(f"AUC (Counterfactual): {auc_cf}")

           
 
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
                    
                    if ((self.step / self.save_and_sample_every) > 0) and self.classification_model_path is not None:
                        self.evaluation_classification_model(trainingLog=trainingLog)
                        print("Evaluated Classification Model Done")
                    
                    with torch.no_grad():
                        milestone = self.step // self.save_and_sample_every
                        batches = num_to_groups(self.num_samples, self.batch_size)
                        classes = torch.tensor([1, 0, 1, 0, 1], dtype=torch.long)
                        classes = classes.to(self.device)
                        all_images_list = []
                        
                        single_noise = torch.randn(1, 1, 8, 2500, device=self.device)
                        expanded_noise = single_noise.expand(self.num_samples, 1, 8, 2500)
                        noise_to_use = expanded_noise
                        
                        for n in batches:
                            batch_classes = classes[:n]
                            images = self.ema.ema_model.sample(batch_classes, cond_scale=self.sampling_cond_scale, noise=noise_to_use[:n])
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