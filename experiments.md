# Experiments of the Diffumatch paper

To work in a zero-shot manner, our paper exploits the [Neural Correspondence Prior](https://arxiv.org/abs/2301.05839) of the common deep functional maps architecture: given XYZ input features, and random weights initialization can be good if the optimization goes well (our contribution!), but then we need [oriented](https://www.lix.polytechnique.fr/~maks/papers/NeurIPS2020_WeakAlign.pdf) datasets as in [Shape Non-rigid Kinematics (SNK)](https://arxiv.org/abs/2403.06804). 

## Downloading datasets 

Some datasets (e.g. TOSCA) are already oriented. We also provide all oriented and non-oriented datasets, you can download them [here](https://huggingface.co/datasets/daidedou/matching_data) and put them into "data" folder.

## Running DiffuMatch over a dataset.

```
python zero_shot.py --dataset dataset_name --config config/matching/sds.yaml
```

Where dataset_name can be FAUST, SCAPE, SHREC19, dtd4dintra, TOSCA.
For the non-isometric datasets (SMAL and DT4D), the best results are obtained with (if you need to apply it on a custom dataset, I recommmend to test out different settings in the notebook):

```
python zero_shot.py --dataset dt4dinter --config config/matching/sds_dt4d.yaml
python zero_shot.py --dataset smalr --config config/matching/sds_smal.yaml
```

We used similar settings as in SNK, where the dataset is non-oriented (we still provide an oriented version, but results tend to be worse), and the shapes are "artificially" aligned via Procrustes using ground-truth correspondences.