import sys
if './' not in sys.path:
	sys.path.append('./')
import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch
def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
device="cuda"

def init_model(sd_weights_path, config_path, output_path):
    #map model
    model = create_model(config_path=config_path)
    scratch_dict = model.state_dict()
    target_dict = {}
    pretrained_weights = torch.load(sd_weights_path,map_location="cpu")
    # print(len(pretrained_weights.keys()))
    pretrained_weights = pretrained_weights['state_dict']
    for sk in scratch_dict.keys():
        if sk in pretrained_weights.keys():
            target_dict[sk] = pretrained_weights[sk].clone()
        else:
            with open('./4.txt','a') as f1:
                f1.write(str(sk)+'\n')
            target_dict[sk] = scratch_dict[sk].clone()
    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
sd_weights_path='./ckpt/sd_1.5_rs_5w_best.ckpt'
config_path='./configs/stable-diffusion/dual/v1-finetune-DIOR.yaml'
output_path='./ckpt/sd15_ini_i.ckpt'
init_model(sd_weights_path, config_path, output_path)
