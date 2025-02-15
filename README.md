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

- **Generate random structures**:
Taking ApolloX/generate_structure/bulk/generate_single_component.py as an example,
you can provide one initial structure, for instance, B_12 Co_12 Fe_12 Mo_12 Ni_12 O_60 with the atoms placed on fixed points. The parameter “file path” is its path for the mother structure and “num_files” is the number of structures you want to generate. In our research, 10000 structures were generated.

```bash
file_path = './POSCAR-ori'
num_files = 10000
shuffle_poscar_lines(file_path, num_files)
```
Run

```bash
python generate_single_component.py
```

- **Acquire pair distribution matrix and cif files**:
Run bulk.py to gain the PDM (pair distribution matrix) information for the structure you generated.
```bash
python bulk.py
```
After run bulk.py, then you will get the pair distribution matrix of the random structures stored in “all_structures_summary.csv”. Note that bulk.py should be placed in the random structures folder.
Then use poscar_to_cif.py to  get cif files of the structures. (Remember to change the path.)
```bash
python ApolloX/prepare_dataset/poscar_to_cif.py
```
- **Standardize the data and split them into training data, test data and validation data**:
Run
```bash
python ApolloX/prepare_dataset/preprocess1.py
```
And you will get the average value and standard deviation in “scaler_stats.txt”. The results are standardized according to the column. 
Data in “train_set_scaled.csv”. “test_set_scaled.csv” and “validation_set_scaled.csv” are in the same form.

- **Add the information of structures(cif) to the training, test and validation data**:
Provide the path of the cif files and run “preprocess2.py”.
```bash
python ApolloX/prepare_dataset/preprocess2.py
```
Make sure the cif foder path is right in preprocess2.py.
```
def read_cif_content(file_name):
    cif_directory = "Apollox/cif"  # make sure the folder path is right
    file_name_with_suffix = os.path.join(cif_directory, file_name + '.cif')
    try:
        with open(file_name_with_suffix, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return 'File not found'
```
after run preprocess2.py you will see the column “cif_file” is added.

- **Convert the training set, test set, and validation set into “.feather” format**:
```bash
python ApolloX/prepare_dataset/preprocess3.py
```

### Train Cond-cdvae model###
cd ~/cond-cdvae-main/conf/data/apollox
In the “.yaml” file, the “root_path” should be changed into the path of “train.feather, test.feather and validation.feather”.
```bash
root_path: /root/cond-cdvae-main/data/10000
prop:
  - pressure
num_targets:
  - 1
niggli: true
primitive: False
graph_method: crystalnn
lattice_scale_method: scale_length
preprocess_workers: 30
readout: mean
max_atoms: 200
otf_graph: false
eval_model_name: calypso
conditions:
  - composition
```

To train the model you run the following (The parameters in the red rectangle determine the path where you can apply the generation model.)

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 nohup python -u /root/cond-cdvae-main/cdvae/run.py  \
model=vae data=apollox/nano project=apollox group=Group_name expname=Exp_name optim.optimizer.lr=1e-4 optim.lr_scheduler.min_lr=1e-5 model.zgivenc.no_mlp=False model.predict_property=False model.encoder.hidden_channels=128 model.encoder.int_emb_size=128 model.encoder.out_emb_channels=128 model.latent_dim=128 model.encoder.num_blocks=4 model.decoder.num_blocks=4 model.conditions.types.pressure.n_basis=80 model.conditions.types.pressure.stop=5 train.pl_trainer.devices=2 +train.pl_trainer.strategy=ddp_find_unused_parameters_true model.prec=32 data.teacher_forcing_max_epoch=60 logging.wandb.mode=online model.cost_lattice=1 > ./apollox_yourname.log 2>&1 &
```
After training,  we can generate a perovskite structured HEA on “~/cond-cdvae-main/log/singlerun/apollox/group/expname” (Substitute them with the parameters in the red rectangle in the picture above):

```bash
CUDA_VISIBLE_DEVICES=2 python ~/cond-cdvae-main/scripts/evaluate.py --model_path `pwd` --tasks gen \
    --formula=Ni110Co110Cu138Bi36La96Sn166 \
    --pressure=0 \
    --label=Ni110Co110Cu138Bi36La96Sn166 \
    --element_values="2.33, 4.94, 5.78, 4.67, 5.00, 8.22, 1.62, 5.16, 6.42, 4.51, 4.51, 7.87, 1.51, 5.12, 6.64, 4.72, 5.32, 7.42, 1.75, 5.17, 6.79, 4.12, 4.10, 8.12, 1.64, 4.51, 6.67, 3.58, 4.76, 7.35, 1.78, 5.22, 6.17, 4.70, 4.87, 6.99" \
    --batch_size=1
```
!!! Replace element_values with the standardized pair distribution matrix of one structure, and several new structures are generated.
The standardized pair distribution matrix can be got in “train_set_scaled.csv”, “test_set_scaled.csv” or “validation_set_scaled.csv”.
Or you can standardize a new pair distribution matrix with the average value and standard deviation in “scaler_stats.txt”.

### Generation and optimization of bulk structures(Folder “PSO”)###
Structures with lower energy can be obtained by the following iteration. Using the PSO algorithm, starting from an initial set of 100 random structures, obtain 60 groups of low-energy pair distribution matrices. Substitute these 60 groups of pair distribution matrices into the generative model to generate 60 new structures. Optimize these 60 structures using DPA-2, then introduce 40 additional random structures to form the next generation’s initial set of 100 structures. Iterate this process repeatedly. 
The number of iterations can be given by the parameter “gen_num” and the number of structures in every iteration is given by the parameter “structure_num_per_gen”. The number of structures after the PSO algorithm is 60% of the number of the input structures.
- **Generate initial structures**
cd in ApolloX/PSO/initial_seeds_and_optimize run:
```bash
./initial_seeds.sh --gen_num 15 --structure_num_per_gen 100

```
“Merged_all_structures_with_energy.csv” is pair distribution matrices and energies of initial structures optimized by the DP model.
“updated_all_structures_summary_batch_1.csv” is pair distribution matrices of initial structures in the first generation of iteration.
In the folder “poscar” are POSCAR files of initial structures. “.optdone.vasp” marks the structure optimized by the DP model.
- **Generate structures using the generative model**
Put all files of pso_and_generating_model and “updated_all_structures_summary_batch_1.csv” obtained in the step above into the folder of using generative model, “~/cond-cdvae-main/log/singlerun/apollox/group/expname”.
In standadize_s.py, revise the value of array “means” and “std_devs”. Open the “scaler_stats.txt”, which is obtained in the process of training the generative model. The array “means” should be replaced by “Train Mean” and the “std_devs” should be replaced by “Train Std”. 
Set the “g” value in the file “gen.sh”. It should be consistent with the current number of generation. For example, “g” is 1 for “updated_all_structures_summary_batch_1.csv” and is “2” for “updated_all_structures_summary_batch_2.csv”.
Run
```bash
./gen.sh
```

Then you will get a folder named “vasp_files_g”. “g” is the number of the current generation. In it are POSCAR files of structures generated by the generative model. 
- **Optimize the structures**
Put the “vasp_files_g” folder into “initial_seeds_and_optimize”. Set the “g” value in the file “optimize.sh”.
Run
```bash
./optimize.sh --structure_num_per_gen 100”.
```
Then you will get “updated_all_structures_summary_batch_2.csv”. It can be used for the next generation of iteration. And optimized structures will be added into the folder “vasp_files_g”.

## Contributing
Contributions to HEA-Gen are welcome. Please submit your pull requests to the repository.

## License
This project is licensed under the  License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This project was supported by [Your Institution or Funding Body]. Special thanks to all collaborators and contributors who have made this project possible.
