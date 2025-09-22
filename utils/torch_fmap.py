import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    #bs, m, n = x.size(0), x.size(1), y.size(1)
    xx = torch.pow(x.squeeze(), 2).sum(1, keepdim=True)
    yy = torch.pow(y.squeeze(), 2).sum(1, keepdim=True).t()
    dist = xx + yy - 2 * torch.inner(x.squeeze(), y.squeeze())
    dist = dist.clamp(min=1e-12).sqrt() 
    return dist

def knnsearch(x, y, alpha=1./0.07, prod=False):
    if prod:
        prods = torch.inner(x.squeeze(), y.squeeze())#/( torch.norm(x.squeeze(), dim=-1)[:, None]*torch.norm(y.squeeze(), dim=-1)[None, :])
        output = F.softmax(alpha*prods, dim=1)
    else:
        distance = euclidean_dist(x, y[None,:])
        output = F.softmax(-alpha*distance, dim=1)
    return output.squeeze()

def extract_p2p_torch(reps_shape, reps_template):
    n_ev = reps_shape.shape[-1]
    with torch.no_grad():
        # print((evecs0_dzo @ fmap01_final.squeeze().T).shape)
        # print(evecs1_dzo.shape)
        reps_shape_torch = torch.from_numpy(reps_shape).float().cuda()
        G_i = LazyTensor(reps_shape_torch[:, None, :].contiguous())  # (M**2, 1, 2)
        reps_template_torch = torch.from_numpy(reps_template).float().cuda()
        X_j = LazyTensor(reps_template_torch[None, :, :n_ev].contiguous())  # (1, N, 2)
        D_ij = ((G_i - X_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances
        indKNN = D_ij.argKmin(1, dim=0).squeeze()  # Grid <-> Samples, (M**2, K) integer tensor
        # pmap10_ref = FM_to_p2p(fmap01_final.detach().squeeze().cpu().numpy(), s_dict['evecs'], template_dict['evecs'])
        # print(indKNN[:10], pmap10_ref[:10])
        indKNN_2 = D_ij.argKmin(1, dim=1).squeeze()
    return indKNN.detach().cpu().numpy(), indKNN_2.detach().cpu().numpy()

def extract_p2p_torch_fmap(fmap_shape_template, evecs_shape, evecs_template):
    n_ev = fmap_shape_template.shape[-1]
    with torch.no_grad():
        # print((evecs0_dzo @ fmap01_final.squeeze().T).shape)
        # print(evecs1_dzo.shape)
        G_i = LazyTensor((evecs_shape[:, :n_ev] @ fmap_shape_template.squeeze().T)[:, None, :].contiguous())  # (M**2, 1, 2)
        X_j = LazyTensor(evecs_template[None, :, :n_ev].contiguous())  # (1, N, 2)
        D_ij = ((G_i - X_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances
        indKNN = D_ij.argKmin(1, dim=0).squeeze()  # Grid <-> Samples, (M**2, K) integer tensor
        # pmap10_ref = FM_to_p2p(fmap01_final.detach().squeeze().cpu().numpy(), s_dict['evecs'], template_dict['evecs'])
        # print(indKNN[:10], pmap10_ref[:10])
        indKNN_2 = D_ij.argKmin(1, dim=1).squeeze()
    return indKNN.detach().cpu().numpy(), indKNN_2.detach().cpu().numpy()

def wlstsq(A, B, w):
    if w is None:
        return torch.linalg.lstsq(A, B).solution
    else:
        assert w.dim() + 1 == A.dim() and w.shape[-1] == A.shape[-2]
        W = torch.diag_embed(w)
        return torch.linalg.lstsq(W @ A, W @ B).solution

def torch_zoomout(evecs0, evecs1, evecs_1_trans, fmap01, target_size, step=1):
    assert fmap01.shape[-2] == fmap01.shape[-1], f"square fmap needed, got {fmap01.shape[-2]} and {fmap01.shape[-1]}"
    fs = fmap01.shape[0]
    for i in range(fs, target_size+1, step):
        indKNN, _ = extract_p2p_torch_fmap(fmap01, evecs0, evecs1)
        #fmap01 = wlstsq(evecs1[..., :i], evecs0[indKNN, :i], None)
        fmap01 = evecs_1_trans[:i, :] @ evecs0[indKNN, :i]
    if fmap01.shape[0] < target_size:
        fmap01 = evecs_1_trans[:target_size, :] @ evecs0[indKNN, :target_size]
    return fmap01