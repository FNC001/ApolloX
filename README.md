# ApolloX V_1.0 (Automatic Prediction by generative mOdel for Large-scaLe Optimization of X-composition materials)

## Overview
ApolloX (Automatic Prediction by generative mOdel for Large-scaLe Optimization of X-composition materials), a physics-guided computational framework designed for the structural prediction and discovery of AHEMs. ApolloX combines a conditional generative deep learning model with particle swarm optimization (PSO), leverag-ing chemical short-range order (CSRO) descriptors encoded as Pair Density Matrices (PDMs). By correlating PDM constraints with enthalpic thermostability, ApolloX effectively narrows the structural search space and bridges the gap between local atomic arrangements and the global energy landscape. Through guided structural generation and optimization, ApolloX successfully manages the disorder inherent in amorphous systems while providing a scalable, energy-driven methodology for multi-component materials.

## Features
- **Generated base on short-range order**: Short-range order alone is sufficient to generate amorphous multi-component structures.
- **Cond-CDVAE Structure Generation**: Employs a conditional generative model to predict feasible crystal structures based on input chemical compositions and desired properties.
- **Material Types**: Supports generation of nano, bulk, and perovskite high-entropy alloys.
- **Maximum atomic number scale**: Accommodate models of high-entropy materials with occupancies at the thousandth percentile.

## System Requirements

- OS: Linux (Ubuntu 20.04+),
- Python Version: 3.9 or later
- GPU: Optional but recommended for deep learning models (e.g., NVIDIA A800 or better)
- CUDA Version: 11.8 (required for GPU acceleration with PyTorch)
- Tested on:
  - Ubuntu 22.04 + CUDA 11.8

Dependencies include:
- torch==2.0.1
- torch_geometric (pyg_lib, torch_scatter, etc.)
- numpy, pandas, scikit-learn


# 1. Installation
> Typical install time: 5–10 minutes on a standard desktop with internet access.
~~~~bash
# Clone the repository
git clone https://github.com/FNC001/ApolloX.git
cd ApolloX/
# Install PyTorch (CUDA 11.8 as an example)
pip install torch==2.0.1 -i https://download.pytorch.org/whl/cu118

# Install PyG dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install other required libraries
pip install -r requirements.txt

# If local development is needed, install in editable mode
pip install -e .
~~~~

---

# 2. Environment Variables

Create or edit the `.env` file in the project root (e.g., using `vi .env`), then `source .env` to load them:

~~~~bash
export PROJECT_ROOT=/path/to/this/project
export HYDRA_JOBS=/path/to/this/project/log
~~~~

---

# 3. Data Preparation

## 3.1 Generate Random Structures

In `ApolloX/generate_structure/bulk/generate_single_component.py`, for example:

~~~~python
file_path = './POSCAR-ori'  # Path to the mother structure file
num_files = 10000           # Number of structures to generate
shuffle_poscar_lines(file_path, num_files)
~~~~

Then run:
~~~~bash
python generate_single_component.py
~~~~

---

## 3.2 Obtain PDM (Pair Distribution Matrix) and CIF Files

1. **Compute PDM**  
   Run in the folder containing the random structures:

   ~~~~bash
   python compute_pdm.py
   ~~~~

   This creates `all_structures_summary.csv` with the PDM of each structure.

2. **Convert POSCAR to CIF**  
   Adjust paths in `transfer_POSCAR_to_cif.py` if necessary, then run:

   ~~~~bash
   python ApolloX/prepare_dataset/poscar_to_cif.py
   ~~~~

   This generates `.cif` files for each POSCAR.

---

## 3.3 Standardize and Split Data

~~~~bash
python ApolloX/prepare_dataset/split_dataset.py
~~~~

This script produces:
- `scaler_stats.txt`: mean and standard deviation for each column  
- `train_set_scaled.csv`, `test_set_scaled.csv`, `validation_set_scaled.csv`: the standardized datasets

---

## 3.4 Append CIF Information to the Data

~~~~bash
python ApolloX/prepare_dataset/append_cif.py
~~~~

In `preprocess2.py`, update `cif_directory` if needed. For example:

~~~~python
def read_cif_content(file_name):
    cif_directory = "ApolloX/cif"  # Change if necessary
    file_name_with_suffix = os.path.join(cif_directory, file_name + '.cif')
    try:
        with open(file_name_with_suffix, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return 'File not found'
~~~~

The resulting CSV files will gain a `cif_file` column with the .cif contents.

---

## 3.5 Convert CSV to Feather

~~~~bash
python ApolloX/prepare_dataset/convert_CVS.py
~~~~

Generates `train.feather`, `test.feather`, `val.feather`.

---

# 4. Training the Cond-CDVAE Model

1. Go to the config folder, for example:

   ~~~~bash
   cd ~/cond-cdvae/conf/data/apollox
   ~~~~

2. Edit the `.yaml` file (e.g., `nano.yaml`) to set `root_path` correctly:

   ~~~~yaml
   root_path: /root/cond-cdvae-main/data  # Example path; change as needed
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
   ~~~~

3. Start training, for example:

   ~~~~bash
   CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 nohup python -u /root/cond-cdvae-main/cdvae/run.py  \
   model=vae data=apollox/mp_apollox project=apollox group=Group_name expname=Exp_name \
   optim.optimizer.lr=1e-4 optim.lr_scheduler.min_lr=1e-5 model.zgivenc.no_mlp=False \
   model.predict_property=False model.encoder.hidden_channels=128 model.encoder.int_emb_size=128 \
   model.encoder.out_emb_channels=128 model.latent_dim=128 model.encoder.num_blocks=4 \
   model.decoder.num_blocks=4 model.conditions.types.pressure.n_basis=80 model.conditions.types.pressure.stop=5 \
   train.pl_trainer.devices=1 +train.pl_trainer.strategy=ddp_find_unused_parameters_true \
   model.prec=32 data.teacher_forcing_max_epoch=60 logging.wandb.mode=online model.cost_lattice=1 \
   > ./apollox_yourname.log 2>&1 &
   ~~~~

4. **Generate new structures**  
   After training, use `evaluate.py`, for example:

   ~~~~bash
   CUDA_VISIBLE_DEVICES=2 python ~/cond-cdvae-main/scripts/evaluate.py \
       --model_path `pwd` \
       --tasks gen \
       --formula=Ni110Co110Cu138Bi36La96Sn166 \
       --pressure=0 \
       --label=Ni110Co110Cu138Bi36La96Sn166 \
       --element_values="2.33, 4.94, 5.78, 4.67, 5.00, 8.22, ..." \
       --batch_size=1
   ~~~~

   - `--element_values` should be the **standardized** PDM (e.g., from `train_set_scaled.csv`).  
   - If you have new PDM data, you can use `scaler_stats.txt` to standardize it.

---

# 5. Batch Generation and Optimization (PSO)

## 5.1 Generate Initial Structures

In `ApolloX/PSO/initial_seeds_and_optimize`, run:

~~~~bash
./initial_seeds.sh --gen_num 15 --structure_num_per_gen 100
~~~~

- **`Merged_all_structures_with_energy.csv`**: PDM plus energies of the initial structures  
- **`updated_all_structures_summary_batch_1.csv`**: the PDM for generation 1  
- **`poscar` folder**: the initial POSCAR files (`.optdone.vasp` indicates DP-optimized)

---

## 5.2 Use the Generative Model to Create New Structures

1. Copy everything from `pso_and_generating_model` plus `updated_all_structures_summary_batch_1.csv` to the generative model folder (e.g., `~/cond-cdvae-main/log/singlerun/apollox/group/expname`).  
2. In `standadize_s.py`, replace `means` and `std_devs` with the `Train Mean` and `Train Std` from `scaler_stats.txt`:

   ~~~~python
   means = [ ... ]     # e.g., [0.123, 0.456, ...]
   std_devs = [ ... ]  # e.g., [0.012, 0.034, ...]
   ~~~~

3. In `gen.sh`, set `g=1` if this is generation 1, then run:

   ~~~~bash
   ./gen.sh
   ~~~~

   This produces a `vasp_files_g` folder (e.g., `vasp_files_1`) containing new POSCARs.

---

## 5.3 Optimize the Generated Structures

1. Copy `vasp_files_g` back to `initial_seeds_and_optimize`.  
2. In `optimize.sh`, set `g=1`, then run:

   ~~~~bash
   ./optimize.sh --structure_num_per_gen 100
   ~~~~

   - Creates `updated_all_structures_summary_batch_2.csv` for the next iteration.  
   - Adds optimized structures to the `vasp_files_g` folder.

Repeat the same procedure for each subsequent generation (incrementing `g`).

---


## Contributing
Contributions to ApolloX are welcome. Please submit your pull requests to the repository.

## License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute this software with proper attribution.  

MIT License

Copyright (c) 2024 Honglin Li & Yongfeng Guo & Xiaoshan Luo, Wanlu Li & Yanchao Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Cite link
https://doi.org/10.48550/arXiv.2503.07043

## Reproducibility

To facilitate reproducibility of our results, we provide all necessary resources on the url "https://doi.org/10.5281/zenodo.15285136". This includes:

- The **training dataset** used in the manuscript (`data/train_set_scaled.csv`, etc.)
- The **trained Cond-CDVAE model** checkpoints used for structure generation and evaluation (`models/cond_cdvae_model.pt` or similar)
- A configuration YAML file (`apollox.yaml`) matching the setup used in the paper

## Acknowledgments

