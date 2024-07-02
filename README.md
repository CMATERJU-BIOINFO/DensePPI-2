# EnMAS: Encoding the Mutation Rates of Amino Acids for Sequence-based PPI Prediction

![Workflow](https://github.com/CMATERJU-BIOINFO/EnMAS/assets/56863228/25884b58-f86d-4bc8-aee7-3b8537a01445)

Identifying interactions between two or more proteins is crucial as it helps understand living organisms' cellular behavior and the underlying molecular mechanisms of various diseases. However, most existing computational algorithms in the field model this as binary interaction between any two proteins, instead of conserving the evolutionary regions of protein function and interactions. This is important for predicting potential interaction sites, vital for drug design, target identification, and understanding disease progression and pathogenic mechanisms. Position-aware encoding provides a way to incorporate the order of amino acids in a protein sequence into the model, thus capturing folding patterns, leading to more accurate predictions of protein structures and, consequently, their interactions. 
This is crucial because the sequence order can affect the structure and function of proteins.

The proposed model, EnMAS, is a novel bio-inspired substitution matrix-based sequence encoding with deep learning for identifying interacting protein pairs. It demonstrates an AUC of 97.13\% on the S. cerevisiae dataset, improving by 1.4\% over the best existing methods. Furthermore, EnMAS outperforms recent sequence-based approaches on the human benchmark dataset, addressing the complexities of protein-protein interaction test classes. EnMAS has been successfully applied for (a) identifying pathogen-host interactions (PHIs) (b) predicting near residue-level interaction even though the model was not trained on residue-level data. The enhanced performance on diverse test sets proves the efficiency of the bio-inspired sequence to image colour encoding strategy using the substitution matrices.

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