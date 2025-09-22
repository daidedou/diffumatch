"""
Simple Gradio app for two-mesh initialization and run phases.
- Upload two meshes (.ply, .obj, .off)
- (Optional) upload a YAML config to override defaults
- Adjust a few numeric settings (sane ranges). Defaults pulled from the provided YAML when present.
- Click **Init** to generate "initialization maps" (here: position/normal-based vertex colors) for both meshes.
- Click **Run** to simulate an iterative evolution with a progress bar, then output another pair of colored meshes.

Replace the bodies of `make_initialization_maps` and `run_evolution` with your real pipeline as needed.

Tested with: gradio >= 4.0, trimesh, pyyaml, numpy.
"""
from __future__ import annotations
import os
import io
import time
import json
import tempfile
from typing import Dict, Tuple, Optional
from omegaconf import OmegaConf
import gradio as gr
import numpy as np
import trimesh
import zero_shot
import yaml
from utils.surfaces import Surface
import notebook_helpers as helper
from utils.meshplot import visu_pts
from utils.torch_fmap import extract_p2p_torch_fmap, torch_zoomout
import torch
import argparse
# -----------------------------
# Utils
# -----------------------------
SUPPORTED_EXTS = {".ply", ".obj", ".off", ".stl", ".glb", ".gltf"}

def _safe_ext(path: str) -> str:
    for ext in SUPPORTED_EXTS:
        if path.lower().endswith(ext):
            return ext
    return os.path.splitext(path)[1].lower()


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    v = vertices.astype(np.float64)
    v = v - v.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(v, axis=1).max()
    if scale == 0:
        scale = 1.0
    v = v / scale
    return v.astype(np.float32)



def ensure_vertex_colors(mesh: trimesh.Trimesh, colors: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    if colors.shape[1] == 3:
        rgba = np.concatenate([colors, 255*np.ones((colors.shape[0],1), dtype=np.uint8)], axis=1)
    else:
        rgba = colors
    out.visual.vertex_colors = rgba
    return out


def export_for_view(surf: Surface, colors: np.ndarray, basename: str, outdir: str) -> Tuple[str, str]:
    """Export to PLY (with vertex colors) and GLB for Model3D preview."""
    glb_path = os.path.join(outdir, f"{basename}.glb")
    mesh = trimesh.Trimesh(surf.vertices, surf.faces, process=False)
    colored_mesh = ensure_vertex_colors(mesh, colors)
    colored_mesh.export(glb_path)
    return glb_path


# -----------------------------
# Algorithm placeholders (replace with your real pipeline)
# -----------------------------
DEFAULT_SETTINGS = {
    "deepfeat_conf.fmap.lambda_": 1,
    "sds_conf.zoomout": 40.0,
    "diffusion.time": 1.0,
    "opt.n_loop": 300,
    "loss.sds": 1.0,
    "loss.proper": 1.0,
}

FLOAT_SLIDERS = {
    # name: (min, max, step)
    "deepfeat_conf.fmap.lambda_": (1e-3, 10.0, 1e-3),
    "sds_conf.zoomout": (1e-3, 10.0, 1e-3),
    "diffusion.time": (1e-3, 10.0, 1e-3),
    "loss.sds": (1e-3, 10.0, 1e-3),
    "loss.proper": (1e-3, 10.0, 1e-3),
}

INT_SLIDERS = {
    "opt.n_loop": (1, 5000, 1),
}


def flatten_yaml_floats(d: Dict, prefix: str = "") -> Dict[str, float]:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten_yaml_floats(v, key))
        elif isinstance(v, (int, float)):
            flat[key] = float(v)
    return flat


def read_yaml_defaults(yaml_path: Optional[str]) -> Dict[str, float]:
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        flat = flatten_yaml_floats(data)
        # Only keep known keys we expose as controls
        defaults = DEFAULT_SETTINGS.copy()
        for k in list(DEFAULT_SETTINGS.keys()):
            if k in flat:
                defaults[k] = float(flat[k])
        return defaults
    return DEFAULT_SETTINGS.copy()




class Datadicts:
    def __init__(self, shape_path, target_path):
        self.shape_path = shape_path
        basename_1 = os.path.basename(shape_path)
        self.shape_dict, _ = helper.load_data(shape_path, "tmp/" + os.path.splitext(basename_1)[0]+".npz", "source", make_cache=True)
        self.shape_surf = Surface(filename=shape_path)
        self.target_path = target_path
        basename_2 = os.path.basename(target_path)
        self.target_dict, _ = helper.load_data(target_path, "tmp/" + os.path.splitext(basename_2)[0]+".npz", "target", make_cache=True)
        self.target_surf = Surface(filename=target_path)
        self.cmap1 = visu_pts(self.shape_surf)

# -----------------------------
# Gradio UI
# -----------------------------
TMP_ROOT = tempfile.mkdtemp(prefix="meshapp_")

def save_array_txt(arr):
    # Create a temporary file with .txt suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as f:
        np.savetxt(f, arr.astype(int), fmt="%d")  # save as text
        return f.name

def build_outputs(surf_a: Surface, surf_b: Surface, cmap_a: np.ndarray, p2p: np.ndarray, tag: str) -> Tuple[str, str, str, str]:
    outdir = os.path.join(TMP_ROOT, tag)
    os.makedirs(outdir, exist_ok=True)
    glb_a = export_for_view(surf_a, cmap_a, f"A_{tag}", outdir)
    cmap_b = cmap_a[p2p]
    glb_b = export_for_view(surf_b, cmap_b, f"B_{tag}", outdir)
    out_file = save_array_txt(p2p)
    return glb_a, glb_b, out_file


def init_clicked(mesh1_path, mesh2_path,
                 lambda_val, zoomout_val, time_val, nloop_val, sds_val, proper_val):
    cfg.deepfeat_conf.fmap.lambda_ = lambda_val
    cfg.sds_conf.zoomout = zoomout_val
    cfg.deepfeat_conf.fmap.diffusion.time = time_val
    cfg.opt.n_loop = nloop_val
    cfg.loss.sds = sds_val
    cfg.loss.proper = proper_val
    matcher.reconf(cfg)
    if not mesh1_path or not mesh2_path:
        raise gr.Error("Please upload both meshes.")
    matcher._init()
    global datadicts
    datadicts = Datadicts(mesh1_path, mesh2_path)

    C12_pred_init, C21_pred_init, feat1, feat2, evecs_trans1, evecs_trans2 = matcher.fmap_model({"shape1": datadicts.shape_dict, "shape2": datadicts.target_dict}, diff_model=matcher.diffusion_model, scale=matcher.fmap_cfg.diffusion.time)
    C12_pred, C12_obj, mask_12 = C12_pred_init
    p2p_init, _ = extract_p2p_torch_fmap(C12_obj, datadicts.shape_dict["evecs"], datadicts.target_dict["evecs"])
    return build_outputs(datadicts.shape_surf, datadicts.target_surf, datadicts.cmap1, p2p_init, tag="init")


def run_clicked(mesh1_path, mesh2_path, yaml_path, lambda_val, zoomout_val, time_val, nloop_val, sds_val, proper_val, progress=gr.Progress(track_tqdm=True)):
    if not mesh1_path or not mesh2_path:
        raise gr.Error("Please upload both meshes.")

    cfg.deepfeat_conf.fmap.lambda_ = lambda_val
    cfg.sds_conf.zoomout = zoomout_val
    cfg.deepfeat_conf.fmap.diffusion.time = time_val
    cfg.opt.n_loop = nloop_val
    cfg.loss.sds = sds_val
    cfg.loss.proper = proper_val
    matcher.reconf(cfg)
    if not mesh1_path or not mesh2_path:
        raise gr.Error("Please upload both meshes.")
    matcher._init()
    global datadicts
    if datadicts is None:
        datadicts = Datadicts(mesh1_path, mesh2_path)
    elif datadicts is not None:
        if not (datadicts.shape_path == mesh1_path and datadicts.target_path == mesh2_path):
            datadicts = Datadicts(mesh1_path, mesh2_path)

    target_normals = torch.from_numpy(datadicts.target_surf.surfel/np.linalg.norm(datadicts.target_surf.surfel, axis=-1, keepdims=True)).float().to("cuda")

    C12_new, p2p, p2p_init, _, loss_save = matcher.optimize(datadicts.shape_dict, datadicts.target_dict, target_normals)
    evecs1, evecs2 = datadicts.shape_dict["evecs"], datadicts.target_dict["evecs"]
    evecs_2trans = evecs2.t() @ torch.diag(datadicts.target_dict["mass"])
    C12_end_zo = torch_zoomout(evecs1, evecs2, evecs_2trans, C12_new.squeeze()[:15, :15], 150)# matcher.cfg.sds_conf.zoomout)
    p2p_zo, _ = extract_p2p_torch_fmap(C12_end_zo, datadicts.shape_dict["evecs"], datadicts.target_dict["evecs"])
    return build_outputs(datadicts.shape_surf, datadicts.target_surf, datadicts.cmap1, p2p_zo, tag="run")


with gr.Blocks(title="DiffuMatch demo") as demo:
    text_in = "Upload two meshes and try our ICCV zero-shot method DiffuMatch! \n"
    text_in += "*Init* will give you a rough correspondence, and you can click on *Run* to see if our method is able to match the two shapes! \n"
    text_in += "*Recommended*: The method requires that the meshes are aligned (rotation-wise) to work well. Also might not work with scans (but try it out!)."
    gr.Markdown(text_in)
    with gr.Row():
        mesh1 = gr.File(label="Mesh A (.ply/.obj/.off)")
        mesh2 = gr.File(label="Mesh B (.ply/.obj/.off)")
        yaml_file = gr.File(label="Optional YAML config", file_types=[".yaml", ".yml"], visible=True)
    # except Exception:
    with gr.Accordion("Settings", open=True):
        with gr.Row():
            lambda_val = gr.Slider(minimum=FLOAT_SLIDERS["deepfeat_conf.fmap.lambda_"][0], maximum=FLOAT_SLIDERS["deepfeat_conf.fmap.lambda_"][1], step=FLOAT_SLIDERS["deepfeat_conf.fmap.lambda_"][2], value=1, label="deepfeat_conf.fmap.lambda_")
            zoomout_val = gr.Slider(minimum=FLOAT_SLIDERS["sds_conf.zoomout"][0], maximum=FLOAT_SLIDERS["sds_conf.zoomout"][1], step=FLOAT_SLIDERS["sds_conf.zoomout"][2], value=40, label="sds_conf.zoomout")
            time_val = gr.Slider(minimum=FLOAT_SLIDERS["diffusion.time"][0], maximum=FLOAT_SLIDERS["diffusion.time"][1], step=FLOAT_SLIDERS["diffusion.time"][2], value=1, label="diffusion.time")
        with gr.Row():
            nloop_val = gr.Slider(minimum=INT_SLIDERS["opt.n_loop"][0], maximum=INT_SLIDERS["opt.n_loop"][1], step=INT_SLIDERS["opt.n_loop"][2], value=300, label="opt.n_loop")
            sds_val = gr.Slider(minimum=FLOAT_SLIDERS["loss.sds"][0], maximum=FLOAT_SLIDERS["loss.sds"][1], step=FLOAT_SLIDERS["loss.sds"][2], value=1, label="loss.sds")
            proper_val = gr.Slider(minimum=FLOAT_SLIDERS["loss.proper"][0], maximum=FLOAT_SLIDERS["loss.proper"][1], step=FLOAT_SLIDERS["loss.proper"][2], value=1, label="loss.proper")
    
    with gr.Row():
        init_btn = gr.Button("Init", variant="primary")
        run_btn = gr.Button("Run", variant="secondary")

    gr.Markdown("### Outputs\nEach stage exports both **GLB** (preview below) and **PLY** (download links) with per‑vertex colors.")
    with gr.Tab("Init"):
        with gr.Row():
            init_view_a = gr.Model3D(label="Shape")
            init_view_b = gr.Model3D(label="Target correspondence (init)")
        with gr.Row():
            out_file_init = gr.File(label="Download correspondences TXT")
    with gr.Tab("Run"):
        with gr.Row():
            run_view_a = gr.Model3D(label="Shape")
            run_view_b = gr.Model3D(label="Target correspondence (run)")
        with gr.Row():
            out_file = gr.File(label="Download correspondences TXT")

    init_btn.click(
        fn=init_clicked,
        inputs=[mesh1, mesh2, lambda_val, zoomout_val, time_val, nloop_val, sds_val, proper_val],
        outputs=[init_view_a, init_view_b, out_file_init],
        api_name="init",
    )

    run_btn.click(
        fn=run_clicked,
        inputs=[mesh1, mesh2, yaml_file, lambda_val, zoomout_val, time_val, nloop_val, sds_val, proper_val],
        outputs=[run_view_a, run_view_b, out_file],
        api_name="run",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the gradio demo")
    parser.add_argument('--config', type=str, default="config/matching/sds.yaml", help='Config file location')
    parser.add_argument('--share', action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    matcher = zero_shot.Matcher(cfg)
    datadicts = None
    demo.launch(share=args.share)
