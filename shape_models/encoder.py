import torch 
import torch.nn as nn
from .layers import DiffusionNet

class Encoder(nn.Module):
    def __init__(self, with_grad=True, key_verts="vertices"):
        super(Encoder, self).__init__()
        self.diff_net = DiffusionNet(
             C_in=3,
             C_out=512,
             C_width=128,
             N_block=4,
             dropout=True,
             with_gradient_features=with_grad,
             with_gradient_rotations=with_grad,
        )
        self.key_verts = key_verts


    def forward(self, shape_dict):
        feats = self.diff_net(shape_dict[self.key_verts], shape_dict["mass"], shape_dict["L"], evals=shape_dict["evals"], 
                               evecs=shape_dict["evecs"], gradX=shape_dict["gradX"], gradY=shape_dict["gradY"], faces=shape_dict["faces"])
        x_out = torch.max(feats, dim=0).values
        return x_out