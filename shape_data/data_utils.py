import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.io import loadmat
import sys
import os
import os.path as osp
import math
import numpy as np
import open3d as o3d
import potpourri3d as pp3d
import torch
from pathlib import Path

class CorrLoader(object):

    def __init__(self, root_dir, data_type='mat'):
        self.root_dir = root_dir
        self.data_type = data_type

    def get_by_names(self, sname0, sname1):
        if self.data_type.endswith('mat'):
            pmap10 = self._load_mat(osp.join(self.root_dir, f'{sname0}-{sname1}.mat'))
            return np.stack((pmap10, np.arange(len(pmap10))), axis=1)
        else:
            raise RuntimeError(f'Data type {self.data_type} is not supported.')

    def _load_mat(self, filepath):
        data = loadmat(filepath)
        pmap10 = np.squeeze(np.asarray(data['pmap10'], dtype=np.int32))
        return pmap10


# https://github.com/RobinMagnet/pyFM/blob/master/pyFM/signatures/HKS_functions.py
def HKS(evals, evects, time_list, scaled=False):
    evals_s = np.asarray(evals).flatten()
    t_list = np.asarray(time_list).flatten()

    coefs = np.exp(-np.outer(t_list, evals_s))
    weighted_evects = evects[None, :, :] * coefs[:, None, :]
    natural_HKS = np.einsum('tnk,nk->nt', weighted_evects, evects)

    if scaled:
        inv_scaling = coefs.sum(1)
        return (1 / inv_scaling)[None, :] * natural_HKS

    else:
        return natural_HKS


def lm_HKS(evals, evects, landmarks, time_list, scaled=False):
    evals_s = np.asarray(evals).flatten()
    t_list = np.asarray(time_list).flatten()

    coefs = np.exp(-np.outer(t_list, evals_s))
    weighted_evects = evects[None, landmarks, :] * coefs[:, None, :]

    landmarks_HKS = np.einsum('tpk,nk->ptn', weighted_evects, evects)

    if scaled:
        inv_scaling = coefs.sum(1)
        landmarks_HKS = (1 / inv_scaling)[None, :, None] * landmarks_HKS

    return landmarks_HKS.reshape(-1, evects.shape[0]).T


def auto_HKS(evals, evects, num_T, landmarks=None, scaled=True):
    abs_ev = sorted(np.abs(evals))
    t_list = np.geomspace(4 * np.log(10) / abs_ev[-1], 4 * np.log(10) / abs_ev[1], num_T)

    if landmarks is None:
        return HKS(abs_ev, evects, t_list, scaled=scaled)
    else:
        return lm_HKS(abs_ev, evects, landmarks, t_list, scaled=scaled)


# https://github.com/RobinMagnet/pyFM/blob/master/pyFM/signatures/WKS_functions.py
def WKS(evals, evects, energy_list, sigma, scaled=False):
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-5)[0].flatten()
    evals = evals[indices]
    evects = evects[:, indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:, None] - np.log(np.abs(evals))[None, :]) / (2 * sigma**2))

    weighted_evects = evects[None, :, :] * coefs[:, None, :]

    natural_WKS = np.einsum('tnk,nk->nt', weighted_evects, evects)

    if scaled:
        inv_scaling = coefs.sum(1)
        return (1 / inv_scaling)[None, :] * natural_WKS

    else:
        return natural_WKS


def lm_WKS(evals, evects, landmarks, energy_list, sigma, scaled=False):
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-2)[0].flatten()
    evals = evals[indices]
    evects = evects[:, indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:, None] - np.log(np.abs(evals))[None, :]) / (2 * sigma**2))
    weighted_evects = evects[None, landmarks, :] * coefs[:, None, :]

    landmarks_WKS = np.einsum('tpk,nk->ptn', weighted_evects, evects)

    if scaled:
        inv_scaling = coefs.sum(1)
        landmarks_WKS = ((1 / inv_scaling)[None, :, None] * landmarks_WKS)

    return landmarks_WKS.reshape(-1, evects.shape[0]).T


def auto_WKS(evals, evects, num_E, landmarks=None, scaled=True):
    abs_ev = sorted(np.abs(evals))

    e_min, e_max = np.log(abs_ev[1]), np.log(abs_ev[-1])
    sigma = 7 * (e_max - e_min) / num_E

    e_min += 2 * sigma
    e_max -= 2 * sigma

    energy_list = np.linspace(e_min, e_max, num_E)

    if landmarks is None:
        return WKS(abs_ev, evects, energy_list, sigma, scaled=scaled)
    else:
        return lm_WKS(abs_ev, evects, landmarks, energy_list, sigma, scaled=scaled)


def compute_hks(evecs, evals, mass, n_descr=100, subsample_step=5, n_eig=35):
    feats = auto_HKS(evals[:n_eig], evecs[:, :n_eig], n_descr, scaled=True)
    feats = feats[:, np.arange(0, feats.shape[1], subsample_step)]
    feats_norm2 = np.einsum('np,np->p', feats, np.expand_dims(mass, 1) * feats).flatten()
    feats /= np.expand_dims(np.sqrt(feats_norm2), 0)
    return feats.astype(np.float32)


def compute_wks(evecs, evals, mass, n_descr=100, subsample_step=5, n_eig=35):
    feats = auto_WKS(evals[:n_eig], evecs[:, :n_eig], n_descr, scaled=True)
    feats = feats[:, np.arange(0, feats.shape[1], subsample_step)]
    feats_norm2 = np.einsum('np,np->p', feats, np.expand_dims(mass, 1) * feats).flatten()
    feats /= np.expand_dims(np.sqrt(feats_norm2), 0)
    return feats.astype(np.float32)


def compute_geodesic_distance(V, F, vindices):
    solver = pp3d.MeshHeatMethodDistanceSolver(np.asarray(V, dtype=np.float32), np.asarray(F, dtype=np.int32))
    dists = [solver.compute_distance(vid)[vindices] for vid in vindices]
    dists = np.stack(dists, axis=0)
    assert dists.ndim == 2
    return dists.astype(np.float32)


def compute_vertex_normals(vertices, faces):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()
    return np.asarray(mesh.vertex_normals, dtype=np.float32)


def compute_surface_area(vertices, faces):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    return mesh.get_surface_area()
 
def numpy_to_open3d_mesh(V, F):
    # Create an empty TriangleMesh object
    mesh = o3d.geometry.TriangleMesh()
    # Set vertices
    mesh.vertices = o3d.utility.Vector3dVector(V)
    # Set triangles
    mesh.triangles = o3d.utility.Vector3iVector(F)
    return mesh


def load_mesh(filepath, scale=True, return_vnormals=False):
    if os.path.splitext(filepath)[1] == ".obj": #Avoid pre process from open3d
        V, F = pp3d.read_mesh(filepath)
        mesh = numpy_to_open3d_mesh(V, F)
    else:
        mesh = o3d.io.read_triangle_mesh(filepath)

    tmat = np.identity(4, dtype=np.float32)
    center = mesh.get_center()
    tmat[:3, 3] = -center
    if scale:
        smat = np.identity(4, dtype=np.float32)
        area = mesh.get_surface_area()
        smat[:3, :3] = np.identity(3, dtype=np.float32) / math.sqrt(area)
        tmat = smat @ tmat
    mesh.transform(tmat)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if return_vnormals:
        mesh.compute_vertex_normals()
        vnormals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        return vertices, faces, vnormals
    else:
        return vertices, faces


def save_mesh(filepath, vertices, faces):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    o3d.io.write_triangle_mesh(filepath, mesh)


def load_geodist(filepath):
    data = loadmat(filepath)
    if 'geodist' in data and 'sqrt_area' in data:
        geodist = np.asarray(data['geodist'], dtype=np.float32)
        sqrt_area = data['sqrt_area'].toarray().flatten()[0]
    elif 'G' in data and 'SQRarea' in data:
        geodist = np.asarray(data['G'], dtype=np.float32)
        sqrt_area = data['SQRarea'].flatten()[0]
    else:
        raise RuntimeError(f'File {filepath} does not have geodesics data.')
    return geodist, sqrt_area


def farthest_point_sampling(points, max_points, random_start=True):
    import torch_cluster

    if torch.is_tensor(points):
        device = points.device
        is_batch = points.dim() == 3
        if not is_batch:
            points = torch.unsqueeze(points, dim=0)
        assert points.dim() == 3

        B, N, D = points.size()
        assert N >= max_points
        bindices = torch.flatten(torch.unsqueeze(torch.arange(B), 1).repeat(1, N)).long().to(device)
        points = torch.reshape(points, (B * N, D)).float()
        sindices = torch_cluster.fps(points, bindices, ratio=float(max_points) / N, random_start=random_start)
        if is_batch:
            sindices = torch.reshape(sindices, (B, max_points)) - torch.unsqueeze(torch.arange(B), 1).long().to(device) * N
    elif isinstance(points, np.ndarray):
        device = torch.device('cpu')
        is_batch = points.ndim == 3
        if not is_batch:
            points = np.expand_dims(points, axis=0)
        assert points.ndim == 3

        B, N, D = points.shape
        assert N >= max_points
        bindices = np.tile(np.expand_dims(np.arange(B), 1), (1, N)).flatten()
        bindices = torch.as_tensor(bindices, device=device).long()
        points = torch.as_tensor(np.reshape(points, (B * N, D)), device=device).float()
        sindices = torch_cluster.fps(points, bindices, ratio=float(max_points) / N, random_start=random_start)
        sindices = sindices.cpu().numpy()
        if is_batch:
            sindices = np.reshape(sindices, (B, max_points)) - np.expand_dims(np.arange(B), 1) * N
    else:
        raise NotImplementedError
    return sindices


def lstsq(A, B):
    assert A.ndim == B.ndim == 2
    sols = scipy.linalg.lstsq(A, B)[0]
    return sols

