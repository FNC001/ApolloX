# Cond-CDVAE

This software implementes Conditional Crystal Diffusion Variational AutoEncoder (Cond-CDVAE), which generates the periodic structure of materials under user-defined chemical compositions and external pressure.

[[Paper]](https://arxiv.org/abs/2403.10846) [[Datasets]](data/)

## Installation

### Install with pip

(torch2.0.1+cu118 for example)

It is suggested to use `conda` (by [conda](https://conda.io/docs/index.html) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)) to create a python>=3.8(3.11 is suggested) environment first, then run the following `pip` commands in this environment.

```bash
pip install torch -i https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -r requirements.txt
pip install -e .
```

### Setting up environment variables

Modify the following environment variables in file by `vi .env`.

- `PROJECT_ROOT`: path to the folder that contains this repo, can get by `pwd`
- `HYDRA_JOBS`: path to a folder to store hydra outputs, if in this repo, git hash can be record by hydra

```env
export PROJECT_ROOT=/path/to/this/project
export HYDRA_JOBS=/path/to/this/project/log
```

## Datasets

All datasets are directly available on `data/` with train/valication/test splits. You don't need to download them again. If you use these datasets, please consider to cite the original papers from which we curate these datasets.

Find more about these datasets by going to our [Datasets](data/) page.

## Training Cond-CDVAE

Example:

```bash
python cdvae/run.py \
    model=vae/vae_nocond \  # vae is default
    project=... group=... expname=... \
    data=... \  # file name without .yml suffix in ./conf/data/
    optim.optimizer.lr=1e-4 optim.lr_scheduler.min_lr=1e-5 \
    data.teacher_forcing_max_epoch=250 data.train_max_epochs=4000 \
    model.beta=0.01 \
    model.fc_num_layers=1 model.latent_dim=... \
    model.hidden_dim=... model.lattice_dropout=... \  # MLP part
    model.hidden_dim=... model.latent_dim=... \
    [model.conditions.cond_dim=...] \
```

For more control options see `./conf`.

To train with multi-gpu:

```bash
CUDA_VISIBLE_DEVICES=0,1 python cdvae/run.py \
    ... \  # can take the same options as before
    train.pl_trainer.devices=2 \
    +train.pl_trainer.strategy=ddp_find_unused_parameters_true
```

Cond-CDVAE uses [hydra](https://hydra.cc) to configure hyperparameters, and users can
modify them with the command line or configure files in `conf/` folder.

After training, model checkpoints can be found in `$HYDRA_JOBS/singlerun/project/group/expname`.

### Training Cond-CDVAE-4M on MP60-CALYPSO

**First to modify `root_path` key in file `conf/data/caly-mp/230617/mp60-B-SiO2+calyhalf2.yaml`**

```bash
# Train
HYDRA_FULL_ERROR=1 nohup python -u cdvae/run.py \
  model=vae data=mp60-CALYPSO/mp60-B-SiO2+calyhalf2 project=cond_cdvae group=mp60-calypso expname=model-4m \
  optim.optimizer.lr=1e-4 optim.lr_scheduler.min_lr=1e-5 model.zgivenc.no_mlp=False model.predict_property=False model.encoder.hidden_channels=128 model.encoder.int_emb_size=128 model.encoder.out_emb_channels=128 model.latent_dim=128 model.encoder.num_blocks=4 model.decoder.num_blocks=4 model.conditions.types.pressure.n_basis=80 model.conditions.types.pressure.stop=5 \
  train.pl_trainer.devices=3 +train.pl_trainer.strategy=ddp_find_unused_parameters_true model.prec=32 \
  data.teacher_forcing_max_epoch=60 > model-4m.log 2>&1 &
```

### Training Cond-CDVAE-86M on MP60-CALYPSO

**Remember to modify `root_path` key in file `conf/data/caly-mp/230617/mp60-B-SiO2+calyhalf2.yaml`**

```bash
# Train
HYDRA_FULL_ERROR=1 nohup python -u cdvae/run.py \
  model=vae data=mp60-CALYPSO/mp60-B-SiO2+calyhalf2 project=cond_cdvae group=mp60-calypso expname=model-86m \
  optim.optimizer.lr=1e-4 optim.lr_scheduler.min_lr=1e-5 model.zgivenc.no_mlp=False model.predict_property=False model.encoder.hidden_channels=512 model.encoder.int_emb_size=256 model.encoder.out_emb_channels=512 model.latent_dim=512 model.encoder.num_blocks=6 model.decoder.hidden_dim=512 model.decoder.num_blocks=6 model.conditions.types.pressure.n_basis=80 model.conditions.types.pressure.stop=5 \
  train.pl_trainer.devices=3 +train.pl_trainer.strategy=ddp_find_unused_parameters_true model.prec=32 \
  data.teacher_forcing_max_epoch=60 > model-86m.log 2>&1 &
```

## Generating materials

To evaluate reconstruction performance:

```bash
python scripts/evaluate.py --model_path MODEL_PATH --tasks recon
```

To generate materials:

```bash
python scripts/evaluate.py --model_path MODEL_PATH --tasks gen \
    [--formula=H2O/--train_data=*.pkl] \
    [--pressure=100] \  # if pressure conditioned
    [--label=xxx] \
    --batch_size=50
```

`MODEL_PATH` will be the path to the trained model. Users can choose one or several of the 3 tasks:

- `recon`: reconstruction, reconstructs all materials in the test data. Outputs can be found in `eval_recon.pt`l
- `gen`: generate new material structures by sampling from the latent space. Outputs can be found in `eval_gen.pt`.
- `opt`: generate new material strucutre by minimizing the trained property in the latent space (requires `model.predict_property=True`). Outputs can be found in `eval_opt.pt`.

`eval_recon.pt`, `eval_gen.pt`, `eval_opt.pt` are pytorch pickles files containing multiple tensors that describes the structures of `M` materials batched together. Each material can have different number of atoms, and we assume there are in total `N` atoms. `num_evals` denote the number of Langevin dynamics we perform for each material.

- `frac_coords`: fractional coordinates of each atom, shape `(num_evals, N, 3)`
- `atom_types`: atomic number of each atom, shape `(num_evals, N)`
- `lengths`: the lengths of the lattice, shape `(num_evals, M, 3)`
- `angles`: the angles of the lattice, shape `(num_evals, M, 3)`
- `num_atoms`: the number of atoms in each material, shape `(num_evals, M)`

## Evaluating model

To compute evaluation metrics, run the following command:

```bash
python scripts/compute_metrics.py --root_path MODEL_PATH --tasks recon gen opt
```

`MODEL_PATH` will be the path to the trained model. All evaluation metrics will be saved in `eval_metrics.json`.

## Results

The trained model checkpoints of Cond-CDVAE-4M are located at [results_archive-v1.0.0/model-4m](https://github.com/ixsluo/cond-cdvae/tree/main/results_archive-v1.0.0/model-4m).

The predicted structures for boron, lithium, and silica are located at [results_archive-v1.0.0/model-4m/mp60+calypso
/eval_gen](https://github.com/ixsluo/cond-cdvae/tree/main/results_archive-v1.0.0/model-4m/mp60%2Bcalypso/eval_gen).


## Authors and acknowledgements

The software is primary written by Xiaoshan Luo based on [CDVAE](https://github.com/txie-93/cdvae).

The GNN codebase and many utility functions are adapted from the [ocp-models](https://github.com/Open-Catalyst-Project/ocp) by the [Open Catalyst Project](https://opencatalystproject.org/). Especially, the GNN implementations of [DimeNet++](https://arxiv.org/abs/2011.14115) and [GemNet](https://arxiv.org/abs/2106.08903) are used.

The main structure of the codebase is built from [NN Template](https://github.com/lucmos/nn-template).

## Citation

Please consider citing the following paper if you find our code & data useful.

```text
Luo, et al., Deep learning generative model for crystal structure prediction, arXiv: 2403.10846, 2024
```

