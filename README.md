# ApolloX V0.1 (HENP structure prediction model)

## Overview
The Alloy Engineering Tool for High-Entropy Research (AET-HER) is a sophisticated computational toolkit designed to facilitate the research and development of novel high-entropy materials. This model is capable of generating high-entropy nanostructures, bulk materials, and perovskite system architectures. It integrates advanced Differential Evolution Entropy Descriptor (DEED) for entropy estimation and the Conditional Crystallographic Discrete Variational Encoder (cond-CDVE) for structural generation.

## Features
- **DEED Entropy Estimation**: Utilizes the Differential Evolution algorithm to accurately estimate the configurational entropy of complex alloys.
- **cond-CDVE Structure Generation**: Employs a conditional generative model to predict feasible crystal structures based on input chemical compositions and desired properties.
- **Material Types**: Supports generation of nano, bulk, and perovskite high-entropy alloys.
- **User-Friendly Interface**: Offers a streamlined command-line interface for easy operation and integration into existing workflows.

## Installation
It is suggested to use conda (by conda or miniconda) to create a python>=3.8(3.11 is suggested) environment first, then run the following pip commands in this environment.
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/FNC001/ApolloX.git
pip install torch==2.0.1 -i https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install -r requirements.txt
pip install -e .
```
## Setting up environment variables
Modify the following environment variables in file by vi .env.

- **PROJECT_ROOT**: path to the folder that contains this repo, can get by pwd
- **HYDRA_JOBS**: path to a folder to store hydra outputs, if in this repo, git hash can be record by hydra
```bash
export PROJECT_ROOT=/path/to/this/project
export HYDRA_JOBS=/path/to/this/project/log
```
## Usage
To start using HEA-Gen, run the following command:
```bash
python hea_gen.py –composition “Fe,Co,Ni,Cu,Al” –type “bulk”
```
Replace the composition and type parameters with your desired specifications.

### Examples
Generate .pkl data on cpu (remeber nohup for slurm):
```bash
 CUDA_VISIBLE_DEVICES=0,1 HYDRA_FULL_ERROR=1 nohup python -u /root/cond-cdvae-main/cdvae/run.py \
  model=vae data=apollox/nano project=apollox group=nano expname=nano_100w_1 \
  optim.optimizer.lr=1e-4 optim.lr_scheduler.min_lr=1e-5 model.zgivenc.no_mlp=False model.predict_property=False model.encoder.hidden_channels=128 model.encoder.int_emb_size=128 model.encoder.out_emb_channels=128 model.latent_dim=128 model.encoder.num_blocks=4 model.decoder.num_blocks=4 model.conditions.types.pressure.n_basis=80 model.conditions.types.pressure.stop=5 \
  train.pl_trainer.devices=2 train.pl_trainer.accelerator=cpu +train.pl_trainer.strategy=ddp_find_unused_parameters_true model.prec=32 \
  data.teacher_forcing_max_epoch=60 \
logging.wandb.mode=offline \
model.cost_lattice=1 \
>  ./apollox_nano.log 2>&1 &
```
lbg for borm
```bash
lbg job submit -i run_mod.json -p ./ -r .
```
Train model on GPU
```bash
CUDA_VISIBLE_DEVICES=0,1,2 HYDRA_FULL_ERROR=1 nohup python -u /root/cond-cdvae-main/cdvae/run.py  \
model=vae data=apollox/nano project=apollox group=nano expname=nano_10w optim.optimizer.lr=1e-4 optim.lr_scheduler.min_lr=1e-5 model.zgivenc.no_mlp=False model.predict_property=False model.encoder.hidden_channels=128 model.encoder.int_emb_size=128 model.encoder.out_emb_channels=128 model.latent_dim=128 model.encoder.num_blocks=4 model.decoder.num_blocks=4 model.conditions.types.pressure.n_basis=80 model.conditions.types.pressure.stop=5 train.pl_trainer.devices=2 +train.pl_trainer.strategy=ddp_find_unused_parameters_true model.prec=32 data.teacher_forcing_max_epoch=60 logging.wandb.mode=online model.cost_lattice=1 > ./apollox_nano.log 2>&1 &
```
Generate a perovskite structured HEA:
```bash
CUDA_VISIBLE_DEVICES=2 python ~/cond-cdvae-main/scripts/evaluate.py --model_path `pwd` --tasks gen \
    --formula=Ni110Co110Cu138Bi36La96Sn166 \
    --pressure=0 \
    --label=Ni110Co110Cu138Bi36La96Sn166 \
    --element_values="2.33, 4.94, 5.78, 4.67, 5.00, 8.22, 1.62, 5.16, 6.42, 4.51, 4.51, 7.87, 1.51, 5.12, 6.64, 4.72, 5.32, 7.42, 1.75, 5.17, 6.79, 4.12, 4.10, 8.12, 1.64, 4.51, 6.67, 3.58, 4.76, 7.35, 1.78, 5.22, 6.17, 4.70, 4.87, 6.99" \
    --batch_size=1
```
## Contributing
Contributions to HEA-Gen are welcome. Please submit your pull requests to the repository.

## License
This project is licensed under the  License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This project was supported by [Your Institution or Funding Body]. Special thanks to all collaborators and contributors who have made this project possible.
