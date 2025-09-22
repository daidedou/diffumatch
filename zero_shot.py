import torch
import numpy as np
import scipy
import os
import sys
import random
import numpy as np
import torch
import time
from datetime import datetime
import importlib
import json
import argparse
from omegaconf import OmegaConf
from snk.loss import PrismRegularizationLoss
from snk.prism_decoder import PrismDecoder
from shape_models.fmap import DFMNet
from shape_models.encoder import Encoder
from diffu_models.losses import VELoss, VPLoss, EDMLoss
from diffu_models.sds import guidance_grad
from utils.torch_fmap import torch_zoomout, knnsearch, extract_p2p_torch_fmap
from utils.utils_func import convert_dict, str_delta, ensure_pretrained_file
from utils.eval import accuracy
from utils.mesh import save_ply, load_mesh
from shape_data import get_data_dirs
from utils.pickle_stuff import safe_load_with_fallback
from utils.geometry import compute_operators, load_operators
from utils.surfaces import Surface
import sys
try:
    import google.colab
    print("Running Colab")
    from tqdm import tqdm
except ImportError:
    print("Running local")
    from tqdm.auto import tqdm


def seed_everything(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

seed_everything()

class Tee:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, message):
        for output in self.outputs:
            output.write(message)
            output.flush()  # ensure it's written immediately

    def flush(self):
        for output in self.outputs:
            output.flush()

class DiffModel:

    def __init__(self, cfg, device="cuda:0"):
        if cfg["train_dir"] == "pretrained":
            url = "https://huggingface.co/daidedou/diffumatch_model/resolve/main/network-snapshot-041216.pkl"
            network_pkl = ensure_pretrained_file(url, "pretrained")
            url_json = "https://huggingface.co/daidedou/diffumatch_model/resolve/main/training_options.json"
            json_filename = ensure_pretrained_file(url_json, "pretrained", filename="training_options.json")
            train_cfg = json.load(open(json_filename))
        else:
            num_exp = cfg["diff_num_exp"]
            files = os.listdir(cfg["train_dir"])
            for file in files:
                if file[:5] == f"{num_exp:05d}":
                    netdir = os.path.join(cfg["train_dir"], file)
            train_cfg = json.load(open(os.path.join(netdir, "training_options.json")))
            pkls = [f for f in os.listdir(netdir) if ".pkl" in f]
            nice_pkls = sorted(pkls, key=lambda x: int(x.split(".")[0].split("-")[-1]))
            chosen_pkl = nice_pkls[-1]
            network_pkl = os.path.join(netdir, chosen_pkl)
        print(f'Loading network from "{network_pkl}"...')
        self.net = safe_load_with_fallback(network_pkl)['ema'].to(device)
        
        print('Done!')
        loss_name = train_cfg['hyper_params']['loss_name']
        self.loss_sde = None
        if loss_name == "EDMLoss":
            self.loss_sde = EDMLoss()
        elif loss_name == "VPLoss":
            self.loss_sde = VPLoss()


class Matcher(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(f'cuda:{cfg["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.diffusion_model = None
        if self.cfg.get("sds", False):
            self.diffusion_model = DiffModel(cfg["sds_conf"])
        self.n_fmap = self.cfg["deepfeat_conf"]["fmap"]["n_fmap"]
        self.n_loop = 0
        if self.cfg.get("optimize", False):
            self.n_loop = self.cfg.opt.get("n_loop", 0)
        self.snk = self.cfg.get("snk", False)
        self.fmap_cfg = self.cfg.deepfeat_conf.fmap
        self.dataloaders = dict()

    def reconf(self, cfg):
        self.cfg = cfg
        self.n_fmap = self.cfg["deepfeat_conf"]["fmap"]["n_fmap"]
        self.n_loop = 0
        if self.cfg.get("optimize", False):
            self.n_loop = self.cfg.opt.get("n_loop", 0)
        self.fmap_cfg = self.cfg.deepfeat_conf.fmap
        self.dataloaders = dict()
        
    def _init(self):
        cfg = self.cfg
        self.fmap_model = DFMNet(self.cfg["deepfeat_conf"]["fmap"]).to(self.device)
        if self.snk:
            self.encoder = Encoder().to(self.device)
            self.decoder = PrismDecoder(dim_in=515).to(self.device)
            self.loss_prism = PrismRegularizationLoss(primo_h=0.02)
            self.soft_p2p = True
            params_to_opt = list(self.fmap_model.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters())
        else:
            params_to_opt = self.fmap_model.parameters()
        self.optim = torch.optim.Adam(params_to_opt, lr=0.001, betas=(0.9, 0.99))
        self.eye = torch.eye(self.n_fmap).float().to(self.device)
        self.eye.requires_grad = False

    def fmap(self, shape_dict, target_dict):
        if self.fmap_cfg.get("use_diff", False):
            C12_pred, C21_pred, feat1, feat2, evecs_trans1, evecs_trans2 = self.fmap_model({"shape1": shape_dict, "shape2": target_dict}, diff_model=self.diffusion_model, scale=self.fmap_cfg.diffusion.time)
            C12_pred, C12_obj, mask_12 = C12_pred
            C21_pred, C21_obj, mask_21 = C21_pred
        else:
            C12_pred, C21_pred, feat1, feat2, evecs_trans1, evecs_trans2 = self.fmap_model({"shape1": shape_dict, "shape2": target_dict})
            C12_obj, C21_obj = C12_pred, C21_pred
            mask_12, mask_21 = None, None
        return C12_pred, C12_obj, C21_pred, C21_obj, feat1, feat2, evecs_trans1, evecs_trans2, mask_12, mask_21
    

    def zo_shot(self, shape_dict, target_dict):
        self._init()
        evecs1, evecs2 = shape_dict["evecs"], target_dict["evecs"]
        _, C12_mask_init, _, _, _, _, _ , _, _, _ = self.fmap(shape_dict, target_dict)
        evecs_2trans = evecs2.t() @ torch.diag(target_dict["mass"])
        new_FM = torch_zoomout(evecs1, evecs2, evecs_2trans, C12_mask_init.squeeze(), self.cfg["zo_shot"])
        indKNN_new, _ = extract_p2p_torch_fmap(new_FM, evecs1, evecs2)
        return new_FM, indKNN_new


    def optimize(self, shape_dict, target_dict, target_normals):
        self._init()
        evecs1, evecs2 = shape_dict["evecs"], target_dict["evecs"]
        C12_pred_init, _, _, _ , _, _, evecs_trans1, evecs_trans2, _, _ = self.fmap(shape_dict, target_dict)
        evecs_2trans = evecs2.t() @ torch.diag(target_dict["mass"])
        evecs_1trans = evecs1.t() @ torch.diag(shape_dict["mass"])
        n_verts_target = target_dict["vertices"].shape[-2]
        
        loss_save = {"cycle": [], "fmap": [], "mse": [], "prism": [], "bij": [], "ortho": [], "sds": [], "lap": [], "proper": []}
        snk_rec = None
        for i in tqdm(range(self.n_loop), "Optimizing matching " + shape_dict['name'] + " " + target_dict['name']):
            C12_pred, C12_obj, C21_pred, C21_obj, feat1, feat2, evecs_trans1, evecs_trans2, _, _ = self.fmap(shape_dict, target_dict)
            if self.cfg.opt.soft_p2p:
                ### A la SNK
                ## P2P 2 -> 1
                soft_p2p_21 = knnsearch(evecs2[:, :self.n_fmap] @ C12_pred.squeeze(), evecs1[:, :self.n_fmap], prod=True)
                C12_new = evecs_trans2[:self.n_fmap, :] @ soft_p2p_21 @ evecs1[:, :self.n_fmap]
                soft_p2p_21 = knnsearch(evecs2[:, :self.n_fmap] @ C12_new.squeeze(), evecs1[:, :self.n_fmap], prod=True)

                ## P2P 1 -> 2 
                soft_p2p_12 = knnsearch(evecs1[:, :self.n_fmap] @ C21_pred.squeeze(), evecs2[:, :self.n_fmap], prod=True)
                C21_new = evecs_trans1[:self.n_fmap, :] @ soft_p2p_12 @ evecs2[:, :self.n_fmap]
                soft_p2p_12 = knnsearch(evecs1[:, :self.n_fmap] @ C21_new.squeeze(), evecs2[:, :self.n_fmap], prod=True)

                l_cycle = ((soft_p2p_12 @ (soft_p2p_21 @ shape_dict["vertices"]) - shape_dict["vertices"])**2).sum(dim=-1).mean()
            else:
                C12_new, C21_new = C12_pred, C21_pred

            l_ortho = ((C12_new.squeeze() @ C12_new.squeeze().T - self.eye)**2).mean() + ((C21_new.squeeze() @ C21_new.squeeze().T - self.eye)**2).mean()
            l_bij = ((C12_new.squeeze() @ C21_new.squeeze() - self.eye)**2).mean() + ((C21_new.squeeze() @ C12_new.squeeze() - self.eye)**2).mean()
            l_lap = ((C12_new @ torch.diag(shape_dict["evals"][:self.n_fmap]) - torch.diag(target_dict["evals"][:self.n_fmap]) @ C12_new)**2).mean()
            l_lap += ((C21_new @ torch.diag(target_dict["evals"][:self.n_fmap]) - torch.diag(shape_dict["evals"][:self.n_fmap]) @ C21_new)**2).mean()


            l_cycle, l_prism, l_mse = torch.as_tensor(0.).float().to(self.device), torch.as_tensor(0.).float().to(self.device), torch.as_tensor(0.).float().to(self.device)
            if self.snk:
                # Latent vector 
                latents = self.encoder(shape_dict)
                latents_duplicate = latents[None, :].repeat(n_verts_target, 1)

                # Prism decoder
                feats_decode = torch.cat((target_dict["vertices"], latents_duplicate), dim=1)
                snk_rec, prism, rots = self.decoder(target_dict, feats_decode)
                l_prism = self.loss_prism(prism, rots, target_dict["vertices"], target_dict["faces"], target_normals)
                l_mse = ((soft_p2p_21 @ shape_dict["vertices"] - snk_rec)**2).sum(dim=-1).mean()
                l_cycle = ((soft_p2p_12 @ (soft_p2p_21 @ shape_dict["vertices"]) - shape_dict["vertices"])**2).sum(dim=-1).mean()
            l_sds, l_proper = torch.as_tensor(0.).float().to(self.device), torch.as_tensor(0.).float().to(self.device)
            if self.fmap_cfg.get("use_diff", False):
                if self.fmap_cfg.diffusion.get("abs", False):
                    C12_in, C21_in = torch.abs(C12_pred).squeeze(), torch.abs(C21_pred).squeeze()
                else:
                    C12_in, C21_in = C12_pred.squeeze(), C21_pred.squeeze()
                grad_12, _ = guidance_grad(C12_in, self.diffusion_model.net, grad_scale=1, batch_size=self.fmap_cfg.diffusion.batch_sds, 
                                           scale_noise=self.fmap_cfg.diffusion.time, device=self.device)
                with torch.no_grad():
                    denoised_12 = C12_pred - self.optim.param_groups[0]['lr'] * grad_12
                targets_12 = torch_zoomout(evecs1, evecs2, evecs_2trans, C12_obj.squeeze(), self.cfg.sds_conf.zoomout)   
                             
                l_proper_12 = ((C12_pred.squeeze()[:self.n_fmap, :self.n_fmap] - targets_12.squeeze()[:self.n_fmap, :self.n_fmap])**2).mean()

                grad_21, _ = guidance_grad(C21_in, self.diffusion_model.net, grad_scale=1, batch_size=self.fmap_cfg.diffusion.batch_sds, 
                                           scale_noise=self.fmap_cfg.diffusion.time, device=self.device)
                #denoised_21 = C21_pred - self.optim.param_groups[0]['lr'] * grad_21
                with torch.no_grad():
                    denoised_21 = C21_pred - self.optim.param_groups[0]['lr'] * grad_21 
                targets_21 = torch_zoomout(evecs2, evecs1, evecs_1trans, C21_obj.squeeze(), self.cfg.sds_conf.zoomout)#, step=10)
                l_proper_21 = ((C21_pred.squeeze()[:self.n_fmap, :self.n_fmap] - targets_21.squeeze()[:self.n_fmap, :self.n_fmap])**2).mean()
                l_proper = l_proper_12 + l_proper_21

                l_sds = ((torch.abs(C12_pred).squeeze()[:self.n_fmap, :self.n_fmap] - denoised_12.squeeze()[:self.n_fmap, :self.n_fmap])**2).mean()
                l_sds += ((torch.abs(C21_pred).squeeze()[:self.n_fmap, :self.n_fmap] - denoised_21.squeeze()[:self.n_fmap, :self.n_fmap])**2).mean()
            loss = torch.as_tensor(0.).float().to(self.device)
            if self.cfg.loss.get("ortho", 0) > 0:
                loss += self.cfg.loss.get("ortho", 0) *  l_ortho
            if self.cfg.loss.get("bij", 0) > 0:
                loss += self.cfg.loss.get("bij", 0) *  l_bij
            if self.cfg.loss.get("lap", 0) > 0:
                loss += self.cfg.loss.get("lap", 0) *  l_lap 
            if self.cfg.loss.get("cycle", 0) > 0:
                loss += self.cfg.loss.get("cycle", 0) *  l_cycle
            if self.cfg.loss.get("mse_rec", 0) > 0:
                loss += self.cfg.loss.get("mse_rec", 0) *  l_mse
            if self.cfg.loss.get("prism_rec", 0) > 0:
                loss += self.cfg.loss.get("prism_rec", 0) *  l_prism
            if self.cfg.loss.get("sds", 0) > 0 and self.fmap_cfg.get("use_diff", False):
                loss += self.cfg.loss.get("sds", 0) * l_sds
            if self.cfg.loss.get("proper", 0) > 0 and self.fmap_cfg.get("use_diff", False):
                loss += self.cfg.loss.get("proper", 0) * l_proper
        
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            loss_save["cycle"].append(l_cycle.item())
            loss_save["ortho"].append(l_ortho.item())
            loss_save["bij"].append(l_bij.item())
            loss_save["sds"].append(l_sds.item())
            loss_save["proper"].append(l_proper.item())
            loss_save["mse"].append(l_mse.item())
            loss_save["prism"].append(l_prism.item())
        indKNN_new_init, _ = extract_p2p_torch_fmap(C12_pred_init, evecs1, evecs2)
        indKNN_new, _ = extract_p2p_torch_fmap(C12_new, evecs1, evecs2)
        return C12_new, indKNN_new, indKNN_new_init, snk_rec, loss_save
    


    def match(self, pair_batch, output_pair, geod_path, refine=True, eval=False):
        shape_dict, _, target_dict, _, target_normals, mapinfo = pair_batch 
        shape_dict_device = convert_dict(shape_dict, self.device)
        target_dict_device = convert_dict(target_dict, self.device)
        print(shape_dict_device["vertices"].device)
        os.makedirs(output_pair, exist_ok=True)


        if self.cfg["optimize"]:
            C12_new, p2p, p2p_init, snk_rec, loss_save = self.optimize(shape_dict_device, target_dict_device, target_normals.to(self.device))
            np.save(os.path.join(output_pair, "p2p_init.npy"), p2p_init)
            np.save(os.path.join(output_pair, "losses.npy"), loss_save)
        else:
            C12_new, p2p = self.zo_shot(shape_dict_device, target_dict_device)
            snk_rec, loss_save = None, None
        np.save(os.path.join(output_pair, "fmap.npy"), C12_new.detach().squeeze().cpu().numpy())
        np.save(os.path.join(output_pair, "p2p.npy"), p2p)
        if snk_rec is not None:
            save_ply(os.path.join(output_pair, "rec.ply"), snk_rec.detach().squeeze().cpu().numpy(), target_dict["faces"])

        if refine:
            evecs1, evecs2 = shape_dict_device["evecs"], target_dict_device["evecs"]
            evecs_2trans = evecs2.t() @ torch.diag(target_dict_device["mass"])
            new_FM = torch_zoomout(evecs1, evecs2, evecs_2trans, C12_new.squeeze(), 128)#, step=10)
            p2p_refined_zo, _ = extract_p2p_torch_fmap(new_FM, evecs1, evecs2)
            np.save(os.path.join(output_pair, "p2p_zo.npy"), p2p)
        if eval:
            file_i, vts_1, vts_2 = mapinfo
            mat_loaded = scipy.io.loadmat(os.path.join(geod_path, file_i + ".mat"))
            A_geod, sqrt_area = mat_loaded['geod_dist'], np.sqrt(mat_loaded['areas_f'].sum())
            _, dist = accuracy(p2p[vts_2], vts_1, A_geod,
                                            sqrt_area=sqrt_area,
                                            return_all=True)
            if refine:
                _, dist_zo = accuracy(p2p_refined_zo[vts_2], vts_1, A_geod,
                                            sqrt_area=sqrt_area,
                                            return_all=True)
                np.savetxt(os.path.join(output_pair, "dists.txt"), (dist.mean(), dist_zo.mean()))
                return p2p, p2p_refined_zo, loss_save, dist.mean(), dist_zo.mean()
            return p2p, loss_save, dist.mean()
        return p2p, loss_save
        



    def _dataset_epoch(self, dataset, name_dataset, save_dir, data_dir):
        os.makedirs(save_dir, exist_ok=True)
        # dloader = DataLoader(dataset, collate_fn=collate_default, batch_size=1)
        num_pairs = len(dataset)
        id_pair = 0
        all_accs = []
        all_accs_zo = []
        t1 = datetime.now()
        save_txt = os.path.join(save_dir, "log.txt")
        # Open a file for writing
        log_file = open(save_txt, 'w')
        # Replace sys.stdout with Tee that writes to both console and file
        sys.stdout = Tee(sys.__stdout__, log_file)

        for batch in dset:
            shape_dict, _, target_dict, _, _, _ = batch
            print("Pair: " + shape_dict['name'] + " " + target_dict['name'])
            name_exp = os.path.join(save_dir, shape_dict['name'], target_dict['name'])
            if self.cfg.get("refine", False):
                _, _, _, dist, dist_zo = self.match(batch, name_exp, os.path.join(data_dir, "geomats", name_dataset), eval=True, refine=True)
            else:
                _, _, dist = self.match(batch, name_exp, os.path.join(data_dir, "geomats", name_dataset), eval=True, refine=False)
            delta = datetime.now() - t1
            fm_delta = str_delta(delta)
            remains = ((delta/(id_pair+1))*num_pairs) - delta
            fm_remains = str_delta(remains)
            all_accs.append(dist)
            accs_mean = np.mean(all_accs)
            if self.cfg.get("refine", False):
                all_accs_zo.append(dist_zo)
                accs_zo = np.mean(all_accs_zo)
                print(f"error: {dist}, zo: {dist_zo}, element {id_pair}/{num_pairs}, mean accuracy: {accs_mean}, mean zo: {accs_zo}, full time: {fm_delta}, remains: {fm_remains}")
            else:
                print(f"error: {dist}, element {id_pair}/{num_pairs}, mean accuracy: {accs_mean}, full time: {fm_delta}, remains: {fm_remains}")
            id_pair += 1
        if self.cfg.get("refine", False):
            print(f"mean error : {np.mean(all_accs)}, mean error refined: {np.mean(all_accs_zo)}")
        else:
            print(f"mean error : {np.mean(all_accs)}")
        sys.stdout = sys.__stdout__ 

    def load_data(self, file, num_evecs=200, make_cache=False, factor=None):
        name = os.path.basename(os.path.splitext(file)[0])
        cache_file = "single_" + name + ".npz"
        verts_shape, faces, vnormals, area_shape, center_shape = load_mesh(file, return_vnormals=True)
        cache_path = os.path.join(self.cfg.cache, cache_file)
        print("Cache is: ", cache_path)
        if not os.path.exists(cache_path) or make_cache:
            print("Computing operators ...")
            compute_operators(verts_shape, faces, vnormals, num_evecs, cache_path, force_save=make_cache)
        data_dict = load_operators(cache_path)
        data_dict['name'] = name
        data_dict_torch = convert_dict(data_dict, self.device)
        #batchify_dict(data_dict_torch)
        return data_dict_torch, area_shape

    def match_files(self, file_shape, file_target):
        batch_shape, _ = self.load_data(file_shape)
        batch_target, _ = self.load_data(file_target) 
        target_surf = Surface(filename=file_target)
        target_normals = torch.from_numpy(target_surf.surfel/np.linalg.norm(target_surf.surfel, axis=-1, keepdims=True)).float().to(self.device)
        batch = batch_shape, None, batch_target, target_normals, None, None
        output_folder = os.path.join(self.cfg.output, batch_shape["name"] + "_" + batch_shape["target"])
        p2p, _ = self.match(batch, output_folder, None)
        return batch_shape, batch_target, p2p




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch the SDS demo over datasets")
    parser.add_argument('--dataset', type=str, default="SCAPE", help='name of the dataset')
    parser.add_argument('--config', type=str, default="config/matching/sds.yaml", help='Config file location')    
    parser.add_argument('--datadir', type=str, default="data", help='path where datasets are store')
    parser.add_argument('--output', type=str, default="results", help="where to store experience results")
    args = parser.parse_args()

    arg_cfg = OmegaConf.from_dotlist(
        [f"{k}={v}" for k, v in vars(args).items() if v is not None]
    )
    yaml_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(yaml_cfg, arg_cfg)
    dataset_name = args.dataset.lower()
    if cfg.get("oriented", False):
        dataset_name += "_ori"
    shape_cls = getattr(importlib.import_module(f'shape_data.{args.dataset.lower()}'), 'ShapeDataset')
    pair_cls = getattr(importlib.import_module(f'shape_data.{args.dataset.lower()}'), 'ShapePairDataset')
    data_dir, name_data_geo, corr_dir = get_data_dirs(args.datadir, dataset_name, 'test')
    name_data_geo = "_".join(name_data_geo.split("_")[:2])
    dset_shape = shape_cls(data_dir, "cache/fmaps", "test", oriented=cfg.get("oriented", False))
    print("Preprocessing shapes done.")
    dset = pair_cls(corr_dir, 'test', dset_shape, rotate=cfg.get("rotate", False))
    exp_time = time.strftime('%y-%m-%d_%H-%M-%S')
    output_logs = os.path.join(args.output, name_data_geo, exp_time)
    matcher = Matcher(cfg)
    matcher._dataset_epoch(dset, name_data_geo, output_logs, args.datadir)