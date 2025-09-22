from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

# feature extractor
from .layers import DiffusionNet
from omegaconf import OmegaConf


def get_mask(evals1, evals2, gamma=0.5, device="cpu"):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
    evals_gamma1, evals_gamma2 = (evals1 ** gamma)[None, :], (evals2 ** gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()

def get_CXX(feat_x, feat_y, evecs_trans_x, evecs_trans_y):
    # compute linear operator matrix representation C1 and C2

    F_hat = torch.bmm(evecs_trans_x, feat_x)
    G_hat = torch.bmm(evecs_trans_y, feat_y)
    A, B = F_hat, G_hat
    
    A_t, B_t = A.transpose(1, 2), B.transpose(1, 2)
    A_A_t, B_B_t = torch.bmm(A, A_t), torch.bmm(B, B_t)
    B_A_t, A_B_t = torch.bmm(B, A_t), torch.bmm(A, B_t)

    C12 = torch.bmm(B_A_t, torch.inverse(A_A_t))

    C21 = torch.bmm(A_B_t, torch.inverse(B_B_t))
    return [C12, C21]

def get_mask_noise(C, diff_model, scale=1, N_est=200, device="cuda", normalize=True, absolute=False):
    with torch.no_grad():
        sig = torch.ones([1, 1, 1, 1], device=device) * scale
        
        noise = torch.randn((N_est, 1, 30, 30), device=device)
        if absolute:
            #noisy_new = torch.abs(C[None, :, :] + noise *scale)
            noisy_new = torch.abs(torch.abs(C[None, :, :]) + noise * scale)
        else:
            noisy_new = C[None, :, :] + noise *scale
        denoised = diff_model.net(noisy_new, sig)
        mask_squared = torch.mean(torch.abs(noisy_new - denoised)/(2*scale), dim=0)/torch.mean(torch.abs(noisy_new), dim=0)
        mask_median = torch.median(mask_squared).item()
        if normalize:
            mask_median = torch.median(mask_squared).item()
            M_denoised = torch.sqrt(torch.clamp(mask_squared-mask_median/2, 0, mask_median*2))
        else:
            M_denoised = torch.sqrt(mask_squared)
    return M_denoised-M_denoised.min()#, mask_median*2)


class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5, use_resolvent=False):
        super().__init__()
        self.lambda_ = lambda_
        self.use_resolvent = use_resolvent
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y, diff_conf=None):
        # compute linear operator matrix representation C1 and C2
        evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
        evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

        F_hat = torch.bmm(evecs_trans_x, feat_x)
        G_hat = torch.bmm(evecs_trans_y, feat_y)
        A, B = F_hat, G_hat



        if diff_conf is not None:
            diff_model, scale, normalize, absolute, N_est = diff_conf
            C12_raw, C21_raw = get_CXX(feat_x, feat_y, evecs_trans_x, evecs_trans_y)
            D12 = get_mask_noise(C12_raw, diff_model, scale=scale, N_est=N_est, normalize=normalize, absolute=absolute)
            D21 = get_mask_noise(C21_raw, diff_model, scale=scale, N_est=N_est, normalize=normalize, absolute=absolute)
        elif self.use_resolvent:
            D12 = get_mask(evals_x.flatten(), evals_y.flatten(), self.resolvant_gamma, feat_x.device).unsqueeze(0)
            D21 = get_mask(evals_y.flatten(), evals_x.flatten(), self.resolvant_gamma, feat_x.device).unsqueeze(0)
        else:
            D12 = (torch.unsqueeze(evals_y, 2) - torch.unsqueeze(evals_x, 1))**2
            D21 = (torch.unsqueeze(evals_x, 2) - torch.unsqueeze(evals_y, 1))**2
        
        A_t, B_t = A.transpose(1, 2), B.transpose(1, 2)
        A_A_t, B_B_t = torch.bmm(A, A_t), torch.bmm(B, B_t)
        B_A_t, A_B_t = torch.bmm(B, A_t), torch.bmm(A, B_t)

        C12_i = []
        for i in range(evals_x.size(1)):
            D12_i = torch.cat([torch.diag(D12[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
            C12 = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D12_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C12_i.append(C12.transpose(1, 2))
        C12 = torch.cat(C12_i, dim=1)

        C21_i = []
        for i in range(evals_y.size(1)):
            D21_i = torch.cat([torch.diag(D21[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_y.size(0))], dim=0)
            C21 = torch.bmm(torch.inverse(B_B_t + self.lambda_ * D21_i), A_B_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C21_i.append(C21.transpose(1, 2))
        C21 = torch.cat(C21_i, dim=1)
        if diff_conf is not None:
            return [C12_raw, C12, D12], [C21_raw, C21, D21]
        else:
            return [C12, C21]


class DFMNet(nn.Module):
    """
    Compilation of the global model :
    - diffusion net as feature extractor
    - fmap + q-fmap
    - unsupervised loss
    """

    def __init__(self, cfg):
        super().__init__()

        # feature extractor #
        with_grad=True
        self.feat = cfg["feat"]

        self.feature_extractor = DiffusionNet(
             C_in=cfg["C_in"],
             C_out=cfg["n_feat"],
             C_width=128,
             N_block=4,
             dropout=True,
             with_gradient_features=with_grad,
             with_gradient_rotations=with_grad,
        )

        # regularized fmap
        self.fmreg_net = RegularizedFMNet(lambda_=cfg["lambda_"],
                                          resolvant_gamma=cfg.get("resolvent_gamma", 0.5), use_resolvent=cfg.get("use_resolvent", False))
        # parameters
        self.n_fmap = cfg["n_fmap"]
        if cfg.get("diffusion", None) is not None:
            self.normalize = cfg["diffusion"]["normalize"]
            self.abs = cfg["diffusion"]["abs"]
            self.N_est = cfg.diffusion.get("batch_mask", 100)

    def forward(self, batch, diff_model=None, scale=1):
        if self.feat == "xyz":
            feat_1, feat_2 = batch["shape1"]["vertices"], batch["shape2"]["vertices"]
        elif self.feat == "wks":
            feat_1, feat_2 = batch["shape1"]["wks"], batch["shape2"]["wks"]
        elif self.feat == "hks":
            feat_1, feat_2 = batch["shape1"]["hks"], batch["shape2"]["hks"]
        else:
            raise Exception("Unknow Feature")


        verts1, faces1, mass1, L1, evals1, evecs1, gradX1, gradY1 = (feat_1, batch["shape1"]["faces"],
                                                                     batch["shape1"]["mass"], batch["shape1"]["L"],
                                                                     batch["shape1"]["evals"], batch["shape1"]["evecs"],
                                                                     batch["shape1"]["gradX"], batch["shape1"]["gradY"])
        verts2, faces2, mass2, L2, evals2, evecs2, gradX2, gradY2 = (feat_2, batch["shape2"]["faces"],
                                                                     batch["shape2"]["mass"], batch["shape2"]["L"],
                                                                     batch["shape2"]["evals"], batch["shape2"]["evecs"],
                                                                     batch["shape2"]["gradX"], batch["shape2"]["gradY"])

        # set features to vertices
        features1, features2 = verts1, verts2
        # print(features1.shape, features2.shape)

        feat1 = self.feature_extractor(features1, mass1, L=L1, evals=evals1, evecs=evecs1,
                                       gradX=gradX1, gradY=gradY1, faces=faces1).unsqueeze(0)
        feat2 = self.feature_extractor(features2, mass2, L=L2, evals=evals2, evecs=evecs2,
                                       gradX=gradX2, gradY=gradY2, faces=faces2).unsqueeze(0)

        evecs_trans1, evecs_trans2 = evecs1.t()[:self.n_fmap] @ torch.diag(mass1.squeeze()), evecs2.squeeze().t()[:self.n_fmap].squeeze() @ torch.diag(mass2.squeeze())
        evals1, evals2 = evals1[:self.n_fmap], evals2[:self.n_fmap]

        #
        if diff_model is not None:
            C12_pred, C21_pred = self.fmreg_net(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2,
                                            diff_conf=[diff_model, scale, self.normalize, self.abs, self.N_est])
        else:
            C12_pred, C21_pred = self.fmreg_net(feat1, feat2, evals1, evals2, evecs_trans1, evecs_trans2)

        return C12_pred, C21_pred, feat1, feat2, evecs_trans1, evecs_trans2