# ApolloX V_1.0 (Automatic Prediction by generative mOdel for Large-scaLe Optimization of X-composition materials)

## Overview
ApolloX (Automatic Prediction by generative mOdel for Large-scaLe Optimization of X-composition materials), a physics-guided computational framework designed for the structural prediction and discovery of AHEMs. ApolloX combines a conditional generative deep learning model with particle swarm optimization (PSO), leverag-ing chemical short-range order (CSRO) descriptors encoded as Pair Density Matrices (PDMs). By correlating PDM constraints with enthalpic thermostability, ApolloX effectively narrows the structural search space and bridges the gap between local atomic arrangements and the global energy landscape. Through guided structural generation and optimization, ApolloX successfully manages the disorder inherent in amorphous systems while providing a scalable, energy-driven methodology for multi-component materials.

## Features
- **Generated base on short-range order**: Short-range order alone is sufficient to generate amorphous multi-component structures.
- **cond-CDVE Structure Generation**: Employs a conditional generative model to predict feasible crystal structures based on input chemical compositions and desired properties.
- **Material Types**: Supports generation of nano, bulk, and perovskite high-entropy alloys.
- **Maximum atomic number scale**: Accommodate models of high-entropy materials with occupancies at the thousandth percentile.

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
## Prepare dataset
The aim of this step is to prepare training data, test data and validation data saved in the form of train.feather, test.feather and val.feather.

- **Generate random structures**:Taking generate_structure/bulk/generate_single_component.py as an example,
you can provide one initial structure, for instance, B_12 Co_12 Fe_12 Mo_12 Ni_12 O_60 with the atoms placed on fixed points. The parameter “file path” is its path and “num_files” is the number of structures you want to generate. In our research, 10000 structures were generated.
<img width="380" alt="image" src="https://github.com/user-attachments/assets/ebf01870-0b54-45e1-9793-7e3d96bb36a2" />

  








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
