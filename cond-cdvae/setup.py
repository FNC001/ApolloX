from setuptools import find_packages, setup

setup(
    name='cdvae',
    version='0.01',
    packages=find_packages(include=['cdvae']),
    install_requires=[
        "ase",
        "click",
        "hydra-core",
        "lightning",
        "pandas",
        "pyarrow",
        "pymatgen",
        "python-dotenv",
        "torch_geometric",
        "torch_scatter",
        "torch_sparse",
        "wandb",
    ],
)
