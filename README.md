# directed-weight

## Environment

```bash
conda create --name dw python=3.8
conda activate dw

# PyTorch & PyTorch Geometric
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install pytorch-geometric -c rusty1s -c conda-forge

# Misc
conda install tqdm scipy scikit-learn pyyaml easydict h5py tensorboard python-lmdb biopython matplotlib nodejs -c conda-forge

# RDKit
# conda install rdkit==2021.03.5 -c conda-forge

# Jupyter
conda install -c conda-forge jupyterlab
conda install -c conda-forge py3dmol
conda install rdkit
pip install atom3d

# Export
conda env export | grep -v "^prefix: " > env.yml
```
