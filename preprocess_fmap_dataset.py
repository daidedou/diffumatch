
from utils.surfaces import Surface, centroid
from utils.geometry import get_operators_small 
from utils.mesh import find_mesh_files
from fmap_data.dfaust_stuff import get_cache_dfaust
import h5py
from tqdm import tqdm
import os 
import itertools
import argparse


## DFAUST utils
def open_sequence(sid, seq, file):
    sidseq = sid + "_" + seq
    if sidseq not in file:
        print('Sequence %s from subject %s not in file' % (seq, sid))
        return None

    verts = file[sidseq][()].transpose([2, 0, 1])
    faces = file['faces'][()]

    return verts, faces


sids  =['50002', '50004', '50007', '50009', '50020',
        '50021', '50022', '50025', '50026', '50027']

pids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
    'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
    'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
    'one_leg_jump', 'running_on_spot']

files = ["registrations_f.hdf5",
            "registrations_m.hdf5"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the computation")
    parser.add_argument('--dataset', type=str, default="DFAUST", help='name of the dataset')
    parser.add_argument('--path', type=str, required=True, help="dataset path")
    parser.add_argument('--n_eig', type=int, default=30, help='Number of eigenbasis components')
    parser.add_argument('--cache_path', type=str, default="data_cache", help="Cache folder")
    args = parser.parse_args()
    num_eigenbasis = args.n_eig
    if args.dataset == "DFAUST":
        with h5py.File(os.path.join(args.path, files[1]), 'r') as f:
            _, faces = open_sequence(sids[0], pids[0], f)
            output_dir = args.cache_path
            os.makedirs(output_dir, exist_ok=True)
            

            saveids = "dfaust.txt"
            for sid, seq in tqdm(list(itertools.product(sids, pids)), "Iterating over sequences"):
                sidseq = sid + "_" + seq
                for fil in files:
                    full_fil = os.path.join(args.path, fil)
                    with h5py.File(full_fil, 'r') as f_read:
                        output = open_sequence(sid, seq, f_read)
                        if output is not None:
                            verts, _ = output
                            for i in tqdm(range(verts.shape[0]), sidseq):
                                cache_i = os.path.join(args.cache_path, get_cache_dfaust(sid, seq, i)) + ".npz"
                                _ = get_operators_small(verts[i], faces, k_eig=args.n_eig, cache_path=cache_i)
                        else:
                            print(f"Sequence : {sidseq} does not exist in {fil} (not an error)!")
    else:
        mesh_files = find_mesh_files(args.path)
        mesh_files.sort()
        surf = Surface(filename=os.path.join(args.path, mesh_files[0]))
        faces = surf.faces
        output_dir = args.cache_path
        os.makedirs(output_dir, exist_ok=True)
        for file_name in tqdm(mesh_files):
            path = os.path.join(output_dir, os.path.splitext(file_name)[0] +  ".npz")
            surf = Surface(filename=os.path.join(args.path, file_name))
            center, sqarea = centroid(surf)
            surf.updateVertices((surf.vertices - center)/sqarea)
            _ = get_operators_small(surf.vertices, surf.faces, k_eig=args.n_eig)