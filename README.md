# COSTA
This repository provides the code for Context-based Statement-Level Vulnerability Localization.

## About dataset

To download the testing dataset used for evaluation in our experiments, run the following commands:

```
gdown https://drive.google.com/uc?id=1ZGIdzKdlzyjX7wSJbP0AfMf5BFovRv1g
```

To download the training and validation dataset used for evaluation in our experiments, run the following commands:

```
gdown https://drive.google.com/uc?id=1dvvZeynTCNdLSBdX7H3wEnRKIZWyILlv
gdow https://drive.google.com/uc?id=11pyuNbkop_5uk10uAoNr4__Tpww65HXb
```

For more information of our dataset, please refer to <a href="https://ieeexplore.ieee.org/document/9796256">LineVul</a> and <a href="https://dl.acm.org/doi/abs/10.1145/3379597.3387501"> Big-Vul.

## Train and test the vulnerability localization models

We provide python source code for training and testing the vulnerability localization models. The source files can be found <a href="https://github.com/ttrangnguyen/COSTA/tree/main/Models">here</a>.
We recommend to use Google Colab to execute the Jupiter notebook COSTA.ipynb.

Please modify hyper-parameters such as batch_size, epoch, vector_length, etc. to fit your own experiments.

## Anlyze source code to obtain contexts
We use <a href="https://github.com/joernio/joern">Joern</a> to analyze source code. The python script for reading CPG nodes and edges can be found <a href="https://github.com/ttrangnguyen/COSTA/blob/main/Joern/joern_script.py">here</a>. 

