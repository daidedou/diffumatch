import torch
import torch.nn as nn
import roma
from shape_models.layers import DiffusionNet


class PrismDecoder(torch.nn.Module):
    def __init__(self, dim_in=1024, dim_out=512, n_width=256, n_block=4, pairwise_dot=True, dropout=False, dot_linear_complex=True, neig=128):
        super().__init__()


        self.diffusion_net = DiffusionNet(
             C_in=dim_in,
             C_out=dim_out,
             C_width=n_width,
             N_block=n_block,
             dropout=dropout,
             with_gradient_features=pairwise_dot,
             with_gradient_rotations=dot_linear_complex,
        )

        self.mlp_refine = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 12),
        )

    def forward(self, batch_dict, latent):
        # original prism
        try:
            verts = batch_dict["vertices"]
        except:
            verts = batch_dict["verts"]
        faces = batch_dict["faces"]
        prism_base = verts[faces]  # (n_faces, 3, 3)
        bs = 1

        # forward through diffusion net
        features = self.diffusion_net(latent, batch_dict["mass"], batch_dict["L"], evals=batch_dict["evals"], 
                               evecs=batch_dict["evecs"], gradX=batch_dict["gradX"], gradY=batch_dict["gradY"], faces=batch_dict["faces"])  # (bs, n_verts, dim)

        # features per face
        x_gather = features.unsqueeze(-1).expand(-1, -1, 3)
        faces_gather = faces.unsqueeze(1).expand(-1, features.shape[-1], -1)
        xf = torch.gather(x_gather, 0, faces_gather)
        features = torch.mean(xf, dim=-1)  # (bs, n_faces, dim)

        # refine features with mlp
        features = self.mlp_refine(features)  # (bs, n_faces, 12)

        # get the translation and rotation
        rotations = features[:, :9].reshape(-1, 3, 3)
        rotations = roma.special_procrustes(rotations)  # (n_faces, 3, 3)
        translations = features[:, 9:].reshape(-1, 3)  # (n_faces, 3)

        # transform the prism
        transformed_prism = (prism_base @ rotations) + translations[:, None]

        # prism to vertices
        features = self.prism_to_vertices(transformed_prism, faces, verts)

        out_features = features.reshape(bs, -1, 3)
        transformed_prism = transformed_prism
        rotations = rotations
        return out_features, transformed_prism, rotations

    def prism_to_vertices(self, prism, faces, verts):
        # initialize the transformed features tensor
        N = verts.shape[0]
        d = prism.shape[-1]
        device = prism.device
        features = torch.zeros((N, d), device=device)

        # scatter the features in K onto L using the indices in F
        features.scatter_add_(0, faces[:, :, None].repeat(1, 1, d).reshape(-1, d), prism.reshape(-1, d))

        # divide each row in the transformed features tensor by the number of faces that the corresponding vertex appears in
        num_faces_per_vertex = torch.zeros(N, dtype=torch.float32, device=device)
        num_faces_per_vertex.index_add_(0, faces.reshape(-1), torch.ones(faces.shape[0] * 3, device=device))
        features /= num_faces_per_vertex.unsqueeze(1).clamp(min=1)

        return features