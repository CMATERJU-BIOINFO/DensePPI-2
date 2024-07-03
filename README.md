# EnMAS: Encoding the Mutation Rates of Amino Acids for Sequence-based PPI Prediction

![Workflow](https://github.com/CMATERJU-BIOINFO/EnMAS/assets/56863228/25884b58-f86d-4bc8-aee7-3b8537a01445)

Understanding interactions between proteins is crucial for elucidating cellular behaviors and the molecular mechanisms underlying various diseases. Accurate prediction of these interactions is vital for drug design, target identification, and understanding disease progression. Most existing computational algorithms model protein interactions as binary relationships, often overlooking the evolutionary regions of protein function and interactions. EnMAS (Enhanced Model for Amino acid Substitution) addresses this gap by providing a novel approach to protein interaction prediction.

EnMAS leverages bio-inspired substitution matrix-based encoding and deep learning to incorporate the order of amino acids in protein sequences for accurate interaction predictions. Demonstrating a 97.13% AUC on the S. cerevisiae dataset, it outperforms existing methods by 1.4%. Its versatility allows for successful application in identifying pathogen-host interactions and near residue-level interaction predictions. EnMAS excels on human benchmark datasets, effectively tackling the complexities of protein-protein interaction test classes, thus providing significant advantages for drug design, target identification, and understanding disease mechanisms.

### Replicating this work

#### To mimic our environment do :

Install Miniconda in your linux machine using this : [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Then, run the following to install our environment.

```bash
  conda env create -f environment.yml -p /path/to/save/
```

#### To get access to this repository and run our code :
Run this on a terminal :
```bash
  git clone https://github.com/CMATERJU-BIOINFO/EnMAS.git
```

Followed by :
```bash
  cd EnMAS/Code/
  conda activate enmasppi
  cd JUPPId/Blosum45/                             #example, change directory according to need
  python3 main_train_test_blosum45JUPPId_C1.py    #example, change filename according to need
```

### Repository contents and directory structure 

To understand and replicate our entire work, the following is a overview of the scripts in different directories. The outputs and untermediates are supposed to be generated at distinct locations only, providing non-conflicting outputs for future use.

```
EnMAS
├── Code                    # Parent folder for all training and testing python codes and .ipynb's
│   ├── input               # Contains input data for test i.e. sCerevisiae, human, SarsCov
│   │   ├── JUPPId
│   │   ├── SarsCov2Data
│   │   └── sCerevisiaeData
│   ├── JUPPId              # Contains scripts to train and test on JUPPIdata(C1, C2, C3), SarsCov2 
│   │   ├── Blosum45
│   │   ├── Blosum90
│   │   ├── DensePPI
│   │   ├── PAM120
│   │   └── PAM250
|   ├── samplePlotsAndInterimResults  # Contains scripts for plot generation and prediction on random data
│   ├── sCerevisiae         # Contains scripts to train and test on sCerevisiae
│   │   ├── Blosum45
│   │   ├── Blosum90
│   │   ├── PAM120
│   │   └── PAM250
│   ├── Utils               # Contains all the necessary utility scripts to run the main train/test scripts
│   └── model.py			# Contains classes and utils for the model used in this study 
├── environment.yml
└── README.md
```

#### NOTE :

All of the experiments were conducted on :
```
  2 x NVIDIA Quadro P5000 16GB
  NVIDIA Driver Version: 545.23.08    
  CUDA Version: 12.3 
  72 core - Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz
  512 GB RAM - 64GBx8 in paralleL
```
