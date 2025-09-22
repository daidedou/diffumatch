import argparse
from pathlib import Path
import os 
from tqdm import tqdm
import potpourri3d as pp3d
import open3d as o3d
import scipy.io as sio
import numpy as np
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shape_data import get_data_dirs

# List of file extensions to consider as "mesh" files.
# Kudos to chatgpt!
# Add or remove extensions here as needed.
MESH_EXTENSIONS = {".ply", ".obj", ".off", ".stl", ".fbx", ".gltf", ".glb"}

def sorted_alphanum(file_list_ordered):
    def convert(text):
        return int(text) if text.isdigit() else text
    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', str(key)) if len(c) > 0]
    return sorted(file_list_ordered, key=alphanum_key)


def list_files(folder_path, name_filter, alphanum_sort=False):
    file_list = [p.name for p in list(Path(folder_path).glob(name_filter))]
    if alphanum_sort:
        return sorted_alphanum(file_list)
    else:
        return sorted(file_list)

def find_mesh_files(directory: Path, extensions: set=MESH_EXTENSIONS, alphanum_sort=False) -> list[Path]:
    """
    Recursively find all files in 'directory' whose suffix (lowercased) is in 'extensions'.
    Returns a list of Path objects.
    """
    matches = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            matches.append(path)
    if alphanum_sort:
        return sorted_alphanum(matches)
    else:
        return sorted(matches)

def save_ply(file_name, V, F, Rho=None, color=None):
    """Save mesh information either as an ASCII ply file.
    https://github.com/emmanuel-hartman/BaRe-ESA/blob/main/utils/input_output.py
    Input:
        - file_name: specified path for saving mesh [string]
        - V: vertices of the triangulated surface [nVx3 numpy ndarray]
        - F: faces of the triangulated surface [nFx3 numpy ndarray]
        - Rho: weights defined on the vertices of the triangulated surface [nVx1 numpy ndarray, default=None]
        - color: colormap [nVx3 numpy ndarray of RGB triples]

    Output:
        - file_name.mat or file_name.ply file containing mesh information
    """

    # Save as .ply file
    nV = V.shape[0]
    nF = F.shape[0]   
    if not ".ply" in file_name:
        file_name += ".ply"
    file = open(file_name, "w")
    lines = ("ply","\n","format ascii 1.0","\n", "element vertex {}".format(nV),"\n","property float x","\n","property float y","\n","property float z","\n")
    
    if color is not None:
        lines += ("property uchar red","\n","property uchar green","\n","property uchar blue","\n")
        if Rho is not None:
            lines += ("property uchar alpha","\n")
    
    lines += ("element face {}".format(nF),"\n","property list uchar int vertex_index","\n","end_header","\n")

    file.writelines(lines)
    lines = []
    for i in range(0,nV):
        for j in range(0,3):
            lines.append(str(V[i][j]))
            lines.append(" ")
        if color is not None:
            for j in range(0,3):
                lines.append(str(color[i][j]))
                lines.append(" ")
            if Rho is not None:
                lines.append(str(Rho[i]))
                lines.append(" ")
                    
        lines.append("\n")
    for i in range(0,nF):
        l = len(F[i,:])
        lines.append(str(l))
        lines.append(" ")

        for j in range(0,l):
            lines.append(str(F[i,j]))
            lines.append(" ")
        lines.append("\n")

    file.writelines(lines)
    file.close()

def numpy_to_open3d_mesh(V, F):
    # Create an empty TriangleMesh object
    mesh = o3d.geometry.TriangleMesh()
    # Set vertices
    mesh.vertices = o3d.utility.Vector3dVector(V)
    # Set triangles
    mesh.triangles = o3d.utility.Vector3iVector(F)
    return mesh



def load_mesh(filepath, scale=True, return_vnormals=False):
    V, F = pp3d.read_mesh(filepath)
    mesh = numpy_to_open3d_mesh(V, F)

    tmat = np.identity(4, dtype=np.float32)
    center = mesh.get_center()
    tmat[:3, 3] = -center
    area = mesh.get_surface_area()
    if scale:
        smat = np.identity(4, dtype=np.float32)
        
        smat[:3, :3] = np.identity(3, dtype=np.float32) / np.sqrt(area)
        tmat = smat @ tmat
    mesh.transform(tmat)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if return_vnormals:
        mesh.compute_vertex_normals()
        vnormals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if scale:
            return vertices, faces, vnormals, area, center
        return vertices, faces, vnormals, area, center
    else:
        return vertices, faces


def mesh_geod_matrix(vertices, faces, do_tqdm=False, verbose=False):
    if verbose:
        print("Setting Geodesic matrix bw vertices")
    n_vertices = vertices.shape[0]
    distmat = np.zeros((n_vertices, n_vertices))
    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    if do_tqdm:
        iterable = tqdm(range(n_vertices))
    else:
        iterable = range(n_vertices)
    for vertind in iterable:
        distmat[vertind] = np.maximum(solver.compute_distance(vertind), 0)
    geod_mat = distmat
    return geod_mat


def prepare_geod_mats(shapes_folder, out, basename=None):
    if basename is None:
        basename = os.path.basename(os.path.dirname(shapes_folder)) #+ "_" + os.path.basename(shapes_folder)
    case = basename
    case_folder_out = os.path.join(out, case)
    os.makedirs(case_folder_out, exist_ok=True)
    all_shapes = [f for f in os.listdir(shapes_folder) if (".ply" in f) or (".off" in f) or ('.obj' in f)]
    for shape in tqdm(all_shapes, "Processing " + os.path.basename(shapes_folder)):
        vertices, faces = pp3d.read_mesh(os.path.join(shapes_folder, shape))
        areas = pp3d.face_areas(vertices, faces)
        mat = mesh_geod_matrix(vertices, faces, verbose=False)
        dict_save = {
            'geod_dist': mat,
            'areas_f': areas
        }
        sio.savemat(os.path.join(case_folder_out, shape[:-4]+'.mat'), dict_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="What to do.")
    parser.add_argument('--make_geods', required=True, type=int, default=0, help='launch computation of geod matrices')
    parser.add_argument('--data', required=False, type=str, default=None)
    parser.add_argument('--datadir', type=str, default="data", help='path where datasets are store')
    parser.add_argument('--basename', required=False, type=str, default=None)
    args = parser.parse_args()
    if args.make_geods:
        # from config import get_geod_path, get_dataset_path, get_template_path
        # output = get_geod_path()
        output = os.path.join(args.datadir, "geomats")
        if args.data == "humans":
            all_folders = [get_data_dirs(args.datadir, "faust", 'test')[0], get_data_dirs(args.datadir, "scape", 'test')[0], get_data_dirs(args.datadir, "shrec19", 'test')[0]]
            for folder in all_folders:
                prepare_geod_mats(folder, output)
        elif args.data == "dt4d":
            data_dir, _, corr_dir = get_data_dirs(args.datadir, args.data, 'test')
            all_folders = sorted([f for f in os.listdir(data_dir) if "cross" not in f])
            for folder in all_folders:
                prepare_geod_mats(os.path.join(data_dir, folder), os.path.join(output, "DT4D"), basename=folder)
        elif args.data is not None:
            data_dir, _, corr_dir = get_data_dirs(args.datadir, args.data, 'test')
            prepare_geod_mats(data_dir, output, args.basename)
    # parser = argparse.ArgumentParser(description="Find all mesh files in a folder and list their paths.")
    # parser.add_argument("--folder", type=Path, help="Path to the folder to search (will search recursively).")
    # args = parser.parse_args()

    # search_folder = args.folder
    # if not search_folder.is_dir():
    #     print(f"Error: '{search_folder}' is not a valid directory.")
    # else:
    #     # Find all matching files
    #     mesh_files = find_mesh_files(search_folder)

    #     # Sort the results for consistency
    #     mesh_files.sort()
    #     for p in mesh_files:
    #         print(p.resolve())