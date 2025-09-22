import torch
from torch.autograd import grad
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F


class WarmupCosineDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, warmup_start_lr=1e-9, max_lr=1e-4, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.last_epoch/self.warmup_steps + (1-self.last_epoch/self.warmup_steps) * self.warmup_start_lr
        else:
            # Cosine decay
            cosine_decay = 0.5 * (1 + np.cos(torch.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            decayed = (1 - self.min_lr / self.max_lr) * cosine_decay + self.min_lr / self.max_lr
            lr = self.max_lr * decayed
        return [lr for _ in self.base_lrs]


def guidance_grad(pred_shape, net, scale_noise, grad_scale=1, batch_size=32, device="cpu", save_guidance_path=None):
    # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
    sigma = 0.01 + torch.rand([batch_size, 1, 1, 1], device=device)*scale_noise
    # predict the noise residual with unet, NO grad!
    with torch.no_grad():
        # sample noise
        noise = torch.randn_like(pred_shape) * sigma
        # pred noise
        x = pred_shape+noise
        denoised = net(x, sigma)
    # w(t), sigma_t^2
    grad = torch.mean(grad_scale * (pred_shape - denoised), dim=0)  # / sigma**2
    #print(sigma.item()**2, weight.item(), torch.norm(pred_shape-denoised).item())
    #print(grad)
    grad = torch.nan_to_num(grad)

    # if save_guidance_path:
    #     with torch.no_grad():
    #         if as_latent:
    #             pred_rgb_512 = self.decode_latents(latents)

    #         # visualize predicted denoised image
    #         # The following block of code is equivalent to `predict_start_from_noise`...
    #         # see zero123_utils.py's version for a simpler implementation.
    #         alphas = self.scheduler.alphas.to(latents)
    #         total_timesteps = self.max_step - self.min_step + 1
    #         index = total_timesteps - t.to(latents.device) - 1 
    #         b = len(noise_pred)
    #         a_t = alphas[index].reshape(b,1,1,1).to(self.device)
    #         sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
    #         sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
    #         pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
    #         result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

    #         # visualize noisier image
    #         result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

    #         # TODO: also denoise all-the-way

    #         # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
    #         viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
    #         save_image(viz_images, save_guidance_path)

    return grad, denoised

def guidance_loss(pred_shape, loss_sde, net, grad_scale=1, device="cpu", save_guidance_path=None):
    grad = guidance_grad(pred_shape, loss_sde, net, grad_scale, device, save_guidance_path)
    targets = (pred_shape - grad).detach()
    loss = 0.5 * F.mse_loss(pred_shape.float(), targets, reduction='sum') / pred_shape.shape[0]
    return loss