import math
from collections import namedtuple
from functools import partial

from numpy import mean
import torch
from torch import autocast, nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat, pack, unpack

from tqdm.auto import tqdm

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        offset_noise_strength = 0.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        use_cfg_plus_plus = False,
        standardize_mean = None,
        standardize_std = None,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond
        
        self.model = model
        self.channels = self.model.channels
        
        self.image_size = image_size
        
        self.objective = objective
        
        
            
        
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'
        self.beta_schedule = beta_schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # use cfg++ when ddim sampling

        self.use_cfg_plus_plus = use_cfg_plus_plus

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        register_buffer('alphas_cumprod_next', F.pad(alphas_cumprod[1:], (0, 1), value = 0.))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)
        
        if standardize_mean is not None and standardize_std is not None:
            register_buffer('standardize_mean', standardize_mean)
            register_buffer('standardize_std', standardize_std)
        
    @property
    def device(self):
        return self.betas.device
    
    def _set_mean_std(self, mean, std):
        standardize_mean = mean.reshape(1, 1, -1, 1) if mean is not None else None
        standardize_std = std.reshape(1, 1, -1, 1) if std is not None else None
        self.register_buffer('standardize_mean', standardize_mean)
        self.register_buffer('standardize_std', standardize_std)
        
    def _normalize(self, x):
        if self.standardize_mean is not None and self.standardize_std is not None:
            return (x - self.standardize_mean) / self.standardize_std
        else:
            raise ValueError("Standardization parameters (mean and std) not set. Call _set_mean_std before normalizing data.")
        
    def _unnormalize(self, x):
        if self.standardize_mean is not None and self.standardize_std is not None:
            return x * self.standardize_std + self.standardize_mean
        else:
            raise ValueError("Standardization parameters (mean and std) not set. Call _set_mean_std before unnormalizing data.")
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    @torch.no_grad()
    def model_predictions(self, x, t, classes, cond_scale = 6., rescaled_phi = 0.7, clip_x_start = False):
        model_output, model_output_null = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output if not self.use_cfg_plus_plus else model_output_null

            x_start = self.predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            x_start_for_pred_noise = x_start if not self.use_cfg_plus_plus else maybe_clip(model_output_null)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)

            x_start_for_pred_noise = x_start
            if self.use_cfg_plus_plus:
                x_start_for_pred_noise = self.predict_start_from_v(x, t, model_output_null)
                x_start_for_pred_noise = maybe_clip(x_start_for_pred_noise)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised = True):
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start
        
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 6., rescaled_phi = 0.7, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, rescaled_phi=rescaled_phi, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi)
        
        img = self._unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale=6., rescaled_phi = 0.7, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        img = torch.randn(shape, device=device)
        
        x_start = None
        
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch, ), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi, clip_x_start=clip_denoised)
            
            if time_next < 0:
                img = x_start
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(img)
            
            img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                        sigma * noise
        
        img = self._unnormalize(img)
        return img
    
    @torch.no_grad()
    def sample(self, classes, cond_scale = 6., rescaled_phi = 0.7):
        batch_size, channels = classes.shape[0], self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, (batch_size, channels, self.image_size[0], self.image_size[1]), cond_scale, rescaled_phi)
    
    @torch.no_grad()
    def ddim_counterfactual_sample_from_clean_to_noisy(self, images, sampling_ratio, cond_scale=6., rescaled_phi = 0.7, clip_denoised = True, progress=True):
        """
            sample x_{t+1}
        """
        batch, device = images.shape[0], images.device
        img = images
        img = self._normalize(img)
        final = [] if progress else None
        sampling_timesteps = self.num_timesteps * sampling_ratio
        
        
        classes = torch.tensor([0] * batch, dtype=torch.long, device=device)
        
        for i in range(int(sampling_timesteps)):
            t = torch.tensor([i] * batch, device=device, dtype=torch.long)
            
            pred_noise, x_start, *_ = self.model_predictions(img, t, classes=classes, cond_scale=0., rescaled_phi=rescaled_phi, clip_x_start=clip_denoised)
            alpha_bar_next = extract(self.alphas_cumprod_next, t, img.shape)
            
            eps = (
                extract(self.sqrt_recip_alphas_cumprod, t, img.shape) * img - x_start
            ) / extract(self.sqrt_recipm1_alphas_cumprod, t, img.shape)
            
            
            mean_pred = (
                x_start * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps
            )
            
            out = {
                "sample": mean_pred, "pred_x_start": x_start, "unnormalized": self._unnormalize(mean_pred)
            }
            
            img = mean_pred
            
            if progress:
                final.append(out)
            else:
                final = out
        
        return (final[-1]["sample"], final) if progress else (final["sample"], [final])
    
    @torch.no_grad()
    def ddim_counterfactual_sample_from_noisy_to_counterfactual(self, images, classes, sampling_ratio, cond_scale=6., rescaled_phi=0.7, clip_denoised=True, progress=True):
        
        batch, device = images.shape[0], images.device
        img = images
        eta = self.ddim_sampling_eta
        final = [] if progress else None
        sampling_timesteps = self.num_timesteps * sampling_ratio
        
        for i in reversed(range(int(sampling_timesteps))):
            t = torch.tensor([i] * batch, device=device)
            
            pred_noise, x_start, *_ = self.model_predictions(img, t, classes=classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi, clip_x_start=clip_denoised)
            
            eps = self.predict_noise_from_start(img, t, x_start)
            
            alpha_bar = extract(self.alphas_cumprod, t, img.shape)
            alpha_bar_prev = extract(self.alphas_cumprod_prev, t, img.shape)
            
            sigma = (
                eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            c = torch.sqrt(1 - alpha_bar_prev - sigma ** 2)
            
            noise = torch.randn_like(img)
            mean_pred = (
                x_start * torch.sqrt(alpha_bar_prev)
                + c * eps
            )
            
            non_zero_mask = (
                (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            )
            
            sample = mean_pred + non_zero_mask * sigma * noise
            
            result_dict = {"sample": sample, "pred_x_start": x_start, "score_mean": pred_noise}
            
            img = sample
            
            if progress:
                final.append(result_dict)
            else:
                final = result_dict
        
        
        return (self._unnormalize(final[-1]["sample"]), final) if progress else (self._unnormalize(final["sample"]), [final])
    
    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def p_losses(self, x_start, t, *, classes, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step

        model_out = self.model(x, t, classes)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    
    def forward(self, img, *args, **kwargs):
        
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        img = self._normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
