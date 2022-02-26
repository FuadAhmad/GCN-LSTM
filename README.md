# GCN-LSTM
**Multivariate Time Series-based Solar Flare Prediction by Functional Network Embedding and Sequence Modeling**

<img width="858" alt="GCN-based node-attributed functional network embedding and LSTM-based local and global sequence embedding" src="https://user-images.githubusercontent.com/11091318/155829709-ad12652c-3bc8-46ff-acdb-71aa3b0ef349.png">
Figure: GCN-based node-attributed functional network embedding and LSTM-based local and global sequence embedding



## Overview
This repository provides the implementation of MVTS-based Solar Flare Prediction using GCN and LSTM. The repository is organised as follows:
- `data/` contains the necessary datasets.
- `models/` contains the implementation of the GCN-LSTM models as well as other baseline implementation.
- `utils/` contains utility functions used in data preparation and in different model implementation.  
- `requirements.txt` contains the libraries and tools needed to execute the models. 


## Dependencies

The models has been tested running under Python 3.9.7, with the required packages installed (see the `requirements.txt` for package details along with their dependencies).
In addition, CUDA 11.1 in a Windows server has been used. 

## Instructions
Step-1:

Install python and other dependencies; 
During installation “Disabled path length limit” if needed.
Optionally can install jupyter notebook to run on jupyter-lab.

Step-2:

Open cmd prompt and navigate to the root folder of the downloaded project where “requirements.txt” exists.
Now execute this command:
pip install -r requirements.txt
It will install all the dependencies with the proper version.

Note 1:
For some cases, you might be required to execute this command before proceeding to the earlier cmd:
pip install wheel
If you are using Anaconda or any other kind of python environment where pip cmd is not available then make sure to install the dependencies with the proper version listed in requirements.txt

Step-3:
Once all dependencies installed,navigate to the extracted project's root directory where main.py exist.
Then you can lunch the project using cmd:jupyter-lab
In jupyter-lab, from the folder directory in models folder, open any .ipynb file then read commented instructions and execute kernels/cells.

