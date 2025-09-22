import numpy as np
import potpourri3d as pp3d 
import torch
import meshplot as mp
from ipygany import PolyMesh, Scene



colors = np.array([[255, 119, 0], #orange
                           [163, 51, 219], #violet
                           [0, 246, 205], #bleu clair
                           [0, 131, 246], #bleu floncé 
                           [246, 234, 0], #jaune
                           [143, 188, 143], #rouge
                    [255, 0, 0]]) 
    

def double_plot_surf(surf_1, surf_2,cmap1=None,cmap2=None):
    d = mp.subplot(surf_1.vertices, surf_1.faces, c=cmap1, s=[2, 2, 0])
    mp.subplot(surf_2.vertices, surf_2.faces, c=cmap2, s=[2, 2, 1], data=d)

def visu_pts(surf, colors=colors, idx_points=None, n_kpts=5):
    areas = np.linalg.norm(surf.surfel, axis=-1, keepdims=True)
    area = np.sqrt(areas.sum()/2)
    if idx_points is None:
        center = (surf.centers*(areas)).sum(axis=0)/areas.sum()
        surf.updateVertices((surf.vertices - center)/area)
        surf.cotanLaplacian()
        idx_points = surf.get_keypoints(n_points=n_kpts)
    solver = pp3d.MeshHeatMethodDistanceSolver(surf.vertices, surf.faces)
    norm_center = solver.compute_distance(idx_points[-1])
    color_array = np.zeros(surf.vertices.shape)
    for i in range(len(idx_points)):
        coeff = 0.1 + 1.5*norm_center[idx_points[i]]
        i_v = idx_points[i]
        dist = solver.compute_distance(i_v)*2
        #color_array += np.clip(1-dist, 0, np.inf)[:, None]*colors[i][None, :]
        color_array += np.exp(-dist**2/coeff)[:, None]*colors[i][None, :]
    color_array = np.clip(color_array, 0, 255.)
    return color_array

def toNP(tens):
    if isinstance(tens, torch.Tensor):
        return tens.detach().squeeze().cpu().numpy()
    return tens


def overlay_surf(shape_vertices, shape_faces, target_vertices, target_faces, colors=["tomato", "darksalmon"]):
    shape_vertices = toNP(shape_vertices)
    target_vertices = toNP(target_vertices)
    mesh_1 = PolyMesh(
        vertices=shape_vertices,
        triangle_indices=shape_faces
    )
    mesh_1.default_color = colors[0]
    
    
    mesh_2 = PolyMesh(
        vertices=target_vertices,
        triangle_indices=target_faces
    )
    mesh_2.default_color = colors[1]
    
    
    scene = Scene([mesh_1, mesh_2])
    #scene = Scene([mesh_5])
    return scene, [mesh_1, mesh_2]