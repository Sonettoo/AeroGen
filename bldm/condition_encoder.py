# The following code is implemented with reference to (GLIGEN)[https://github.com/gligen/GLIGEN/tree/master]

import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierEmbedder:
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)

class BoxEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()  # Correct super() call
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy 

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, boxes, masks, text_embeddings):

        # print(len(boxes))
        # if isinstance(boxes,list):
        boxes=boxes[0]
        masks=masks[0]
        text_embeddings=text_embeddings[0]
        # print(masks.shape)
        # print(boxes.shape)
        # print(text_embeddings.shape)
        # masks = merge_first_two_dims(masks)
        # boxes = merge_first_two_dims(boxes)
        # text_embeddings = merge_first_two_dims(text_embeddings)
        # print(masks.shape)
        # print(boxes.shape)
        # print(text_embeddings.shape)
        B, N, _ = boxes.shape 
        masks = masks.unsqueeze(-1)

        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 --> B*N*C

        text_null = self.null_text_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)

        text_embeddings = text_embeddings * masks + (1 - masks) * text_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null
        # print(text_embeddings.shape)
        # print(xyxy_embedding.shape)

        objs = self.linears(torch.cat([text_embeddings, xyxy_embedding], dim=-1))
        assert objs.shape == torch.Size([B, N, self.out_dim])        
        return objs
    
def merge_first_two_dims(tensor):
    """
    """
    shape = tensor.shape
    new_shape = (shape[0] * shape[1],) + shape[2:]
    return tensor.reshape(new_shape)

class RBoxEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super(RBoxEncoder, self).__init__()  # Correct super() call
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 8  # 2 is sin&cos, 8 is new input dim

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, boxes, masks, text_embeddings):
        boxes = boxes[0]
        masks = masks[0]
        text_embeddings = text_embeddings[0]

        B, N, _ = boxes.shape 
        masks = masks.unsqueeze(-1)

        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*8 --> B*N*C

        text_null = self.null_text_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)

        text_embeddings = text_embeddings * masks + (1 - masks) * text_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        objs = self.linears(torch.cat([text_embeddings, xyxy_embedding], dim=-1))
        assert objs.shape == torch.Size([B, N, self.out_dim])        
        return objs
