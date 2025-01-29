# From Mean to Extreme: Formal Differential Privacy Bounds on the Success of Real-World Data Reconstruction Attacks

Accompanying code for the paper "From Mean to Extreme: Formal Differential Privacy Bounds on the Success of Real-World Data Reconstruction Attacks". 

## Setup
 1. Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).
 2. `conda env create -f environment.yaml`
 3. `conda activate frommeantoextreme`
 4. Copy the Imagenet dataset (ILSVRC2012) into the data folder.

## How to reproduce figures
[figures.py](figures.py) contains the code to reproduce all figures in the paper. 

We recommend using the [python interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py) provided by vscode, which allows to execute and inspect the output for each figure at a time. 
However, running `python figures.py` will also save all figures to the working directory. 