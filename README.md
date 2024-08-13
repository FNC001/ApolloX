# Alloy Engineering Tool for High-Entropy Research (AET-HER)

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
Generate a high-entropy nanostructure:
```bash
python hea_gen.py –composition “Ti,Zr,Hf,V,Nb” –type “nano”
```
Generate a perovskite structured HEA:
```bash
python hea_gen.py –composition “Ca,Ti,O,N” –type “perovskite”
```
## Contributing
Contributions to HEA-Gen are welcome. Please submit your pull requests to the repository.

## License
This project is licensed under the  License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This project was supported by [Your Institution or Funding Body]. Special thanks to all collaborators and contributors who have made this project possible.
