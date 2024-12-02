# The following code is implemented with reference to (ControlNet)[https://github.com/lllyasviel/ControlNet]

import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
import numpy as np
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, log_bbox_as_image
from ldm.models.diffusion.ddim import DDIMSampler

class BLDM(LatentDiffusion):
    def __init__(self,condition_tokenizer,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_tokenizer = instantiate_from_config(condition_tokenizer)
        # self.mask_attention = instantiate_from_config(mask_attention)
        self.control_scales = [1.0] * 13
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        if len(batch['bboxes']) !=0:
            bboxes=batch['bboxes']
            if bs is not None:
                bboxes=bboxes[:bs]
            bboxes=bboxes.to(self.device)
            bboxes = bboxes.to(memory_format=torch.contiguous_format).float()
        else:
            bboxes=torch.zeros(1,1).to(self.device).to(memory_format=torch.contiguous_format).float()


        if len(batch['category_conditions']) !=0:
            category_conditions=batch['category_conditions']
            if bs is not None:
                category_conditions=category_conditions[:bs]
            category_conditions=category_conditions.to(self.device)
            category_conditions = category_conditions.to(memory_format=torch.contiguous_format).float()
        else:
            category_conditions=torch.zeros(1,1).to(self.device).to(memory_format=torch.contiguous_format).float()
        
        if len(batch['mask_conditions']) != 0:
            mask_conditions = batch['mask_conditions']
            if bs is not None:
                mask_conditions = mask_conditions[:bs]
            mask_conditions = mask_conditions.to(self.device).to(memory_format=torch.contiguous_format).float()
        else:
            mask_conditions = torch.zeros(1,1,1).to(self.device).to(memory_format=torch.contiguous_format).float()

        if len(batch['mask_vector']) != 0:
            mask_vector = batch['mask_vector']
            if bs is not None:
                mask_vector = mask_vector[:bs]
            mask_vector = mask_vector.to(self.device).to(memory_format=torch.contiguous_format).float()
        else:
            mask_vector = torch.zeros(1,1,1).to(self.device).to(memory_format=torch.contiguous_format).float()

        return x, dict(c_crossattn=[c], bbox_control=[bboxes], category_control=[category_conditions],mask_control=[mask_conditions],mask_vector=[mask_vector])
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        category_control=cond['category_control']
        mask_control=cond['mask_control']
        bbox_control=cond['bbox_control']
        mask_vector=cond['mask_vector']

        if isinstance(category_control[0],list):
            category_control=category_control[0]
        # control = self.condition_tokenizer(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), image_control=image_control, mask_control=mask_control, timesteps=t, context=cond_txt)
        control = self.condition_tokenizer(text_embeddings=category_control,masks=mask_vector,boxes=bbox_control)
        # no cut
        # cond_txt=torch.cat((cond_txt,control),dim=1)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, category_control=category_control, mask_control=mask_control
                            )
        # print(eps.shape)
        
            #image_control=image_control,mask_control=mask_control
        
        # if classifier is not None:
        #     a=1
        # print(eps)

        return eps
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        category_control=c['category_control'][:N]
        bbox_control=c["bbox_control"][0][:N]
        mask_control=c["mask_control"][0][:N]
        mask_vector=c["mask_vector"][0][:N]

        c =  c["c_crossattn"][0][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        # log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        log["bbox_images"] = log_bbox_as_image(bbox_control)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise rowmask_vector
            samples, z_denoise_row = self.sample_log(cond={"c_crossattn": [c],"bbox_control":[bbox_control],
                                                    "category_control":[category_control],"mask_control":[mask_control],"mask_vector":[mask_vector]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            # ubbox_control=torch.zeros_like(bbox_control)

            uc_full = {"c_crossattn": [uc_cross],"category_control":[category_control], 
                       "bbox_control":[bbox_control],"mask_control":[mask_control],"mask_vector":[mask_vector]}
            samples_cfg, _ = self.sample_log(cond={"c_crossattn": [c],"category_control":[category_control],
                                            "bbox_control":[bbox_control],"mask_control":[mask_control], "mask_vector":[mask_vector]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = (self.channels, self.image_size, self.image_size)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        trainable_names=[]
        model_name=[]
        lr = self.learning_rate
        params = list(self.condition_tokenizer.parameters())
        # for nn in self.condition_tokenizer.named_parameters():

        #     print(nn)
        # params = list(self.mask_attention.parameters())
        for name, p in self.model.named_parameters():
            model_name.append(name)
            # if "transformer_blocks" in name and 'attn1' not in name and 'transformer_blocks.0.norm1' not in name:
            if "transformer_blocks" in name:
                # New added Attention layers 
                params.append(p) 
                trainable_names.append(name)
            if 'maskcrossattention' in name:
                params.append(p) 
                trainable_names.append(name)
        # if not self.sd_locked:
        #     params += list(self.model.diffusion_model.output_blocks.parameters())
        #     params += list(self.model.diffusion_model.out.parameters())
        # with open('./1.txt','a') as f:
        #     for ns in trainable_names:
        #         f.write(ns+'\n')
        # with open('./2.txt','a') as f:
        #     for ns in model_name:
        #         f.write(ns+'\n')
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def low_vram_shift(self, is_diffusing): # not use
        if is_diffusing:
            self.model = self.model.cuda()
            self.condition_tokenizer = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.condition_tokenizer = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()