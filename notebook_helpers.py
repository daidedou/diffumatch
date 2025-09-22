from utils.mesh import load_mesh
from utils.geometry import get_operators, load_operators
import os
from utils.utils_func import convert_dict
from utils.surfaces import Surface
import numpy as np


device = "cuda:0"

def load_data(file, cache_path, name, num_evecs=128, make_cache=False, factor=None):
    if factor is None:
        verts_shape, faces, vnormals, area_shape, center_shape = load_mesh(file, return_vnormals=True)
    else:
        verts_shape, faces, vnormals, area_shape, center_shape = load_mesh(file, return_vnormals=True, scale=False)
        verts_shape = verts_shape/factor
        area_shape /= factor**2
    # print("Cache is: ", cache_path)
    if not os.path.exists(cache_path) or make_cache:
        print("Computing operators ...")
        get_operators(verts_shape, faces, num_evecs, cache_path, vnormals)
    data_dict = load_operators(cache_path)
    data_dict['name'] = name
    data_dict['normals'] = vnormals
    data_dict['vertices'] = verts_shape
    data_dict_torch = convert_dict(data_dict, device)
    #batchify_dict(data_dict_torch)
    return data_dict_torch, area_shape

def get_map_info(file_1, file_2, dict_1, dict_2, dataset):
    shape_dict, target_dict = dict_1, dict_2
    name_1, name_2 = shape_dict["name"], target_dict["name"]
    if dataset is None:
        vts_1, vts_2 = np.arange(shape_dict['vertices'].shape[0]), np.arange(target_dict['vertices'].shape[0])
        map_info = (file_1, file_2, vts_1, vts_2)
    file_vts_1 = file_1.replace("shapes", "correspondences")[:-4] + ".vts"
    vts_1 = np.loadtxt(file_vts_1).astype(np.int32) - 1

    file_vts_2 = file_2.replace("shapes", "correspondences")[:-4] + ".vts"
    vts_2 = np.loadtxt(file_vts_2).astype(np.int32) - 1
    if "DT4D" in dataset:    
        file_vts_1 = file_1.replace("shapes", "correspondences")[:-4] + ".vts"
        vts_1 = np.loadtxt(file_vts_1).astype(np.int32) - 1
    
        file_vts_2 = file_2.replace("shapes", "correspondences")[:-4] + ".vts"
        vts_2 = np.loadtxt(file_vts_2).astype(np.int32) - 1
        if name_1 == name_2:
            map_info = (file_1, file_2, vts_1, vts_2)
        elif ("crypto" in name_1) or ("crypto" in name_2):
            name_cat_1, name_cat_2 = name_1.split(os.sep)[0], name_2.split(os.sep)[0]
            data_path = os.path.dirname(os.path.dirname(os.path.dirname(file_1)))
            map_file = os.path.join(data_path, "correspondences/cross_category_corres", f"{name_cat_1}_{name_cat_2}.vts")
            if os.path.exists(map_file):
                map_idx = np.loadtxt(map_file).astype(np.int32) - 1
                map_info = (file_1, file_2, vts_1, vts_2[map_idx])
        else:
            print("NO GROUND TRUTH PAIRS")
    else:
        map_info = (file_1,  file_2, vts_1, vts_2)
    return map_info


def load_pair(cache, id_1, id_2, name_1, name_2, dataset):
    if "SCAPE" in dataset:
        os.makedirs(cache, exist_ok=True) 
        cache_file = os.path.join(cache, f"mesh{id_1:03d}_mesh_256k_0n.npz")
        shape_file = f"data/{dataset}/shapes/mesh{id_1:03d}.ply"
        shape_surf = Surface(filename=shape_file)
        shape_dict, _  = load_data(shape_file, cache_file, str(id_1))
        
        cache_file = os.path.join(cache, f"mesh{id_2:03d}_mesh_256k_0n.npz")
        target_file = f"data/{dataset}/shapes/mesh{id_2:03d}.ply"
        target_surf = Surface(filename=target_file)
        target_dict, _  = load_data(target_file, cache_file, str(id_2))
        map_info = get_map_info(shape_file, target_file, shape_dict, target_dict, "SCAPE")
    elif "DT4D" in dataset:
        cache_file = os.path.join(cache, f"{name_1}_mesh_256k_0n.npz")
        shape_file = f"data/DT4D_r_ori/shapes/{name_1}.ply"
        shape_surf = Surface(filename=shape_file)
        shape_dict, _  = load_data(shape_file, cache_file, name_1)
        cache_file = os.path.join(cache, f"{name_2}_mesh_256k_0n.npz")
        # cache_file = f"../nonrigiddiff/cache/snk/{name_2}.npz"
        cache_file = f"../nonrigiddiff/cache/attentive/{name_2}_mesh_256k_0n.npz"
        target_file = f"data/DT4D_r_ori/shapes/{name_2}.ply"
        target_surf = Surface(filename=target_file)
        target_dict, _  = load_data(target_file, cache_file, name_2)
        map_info = get_map_info(shape_file, target_file, shape_dict, target_dict, "DT4D")
    return shape_surf, target_surf, shape_dict, target_dict, map_info