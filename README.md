# DensePPI-2: A bio-inspired update for sequence based PPI prediction leveraging mutation rates

![Work](https://github.com/user-attachments/assets/1109440a-7928-47b4-8009-0f33a6d1a6ab)


Understanding interactions between proteins is crucial for elucidating cellular behaviors and the molecular mechanisms underlying various diseases. Accurate prediction of these interactions is vital for drug design, target identification, and understanding of disease progression. Most existing computational algorithms model protein interactions as binary relationships, often overlooking the evolutionary regions of protein function and interactions. _**DensePPI-2**_ (Encoding the Mutation Rates of Amino Acids for Sequence-based PPI Prediction) addresses this gap by providing a novel approach to protein interaction prediction.

_**DensePPI-2**_ leverages bio-inspired substitution matrix-based encoding and deep learning to incorporate the order of amino acids in protein sequences for accurate interaction predictions. Demonstrating a 97.13% AUC on the S.cerevisiae dataset, it outperforms existing methods by 1.4%. Its versatility allows for successful application in identifying pathogen-host interactions and near residue-level interaction predictions. _**DensePPI-2**_ excels on human benchmark datasets, effectively tackling the complexities of protein-protein interaction test classes, thus providing significant advantages for drug design, target identification, and understanding disease mechanisms.

--------------------------------------------------------------------------------------------------------------------------------

To get started with _**DensePPI-2**_, follow the instructions in the Installation Guide and explore the Usage Examples to see how DensePPI-2 can be applied to protein interaction prediction tasks.

--------------------------------------------------------------------------------------------------------------------------------

### Installation Guide: (Replicating this work)

#### Setting Up the Environment for _**DensePPI-2**_ :

Install Miniconda in your linux machine using this : [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Then, run the following to install our environment.

```bash
  conda env create -f environment.yml -p /path/to/save/
```

#### To get access to this repository and run our code :
Run this on a terminal :
```bash
  git clone https://github.com/CMATERJU-BIOINFO/DensePPI-2.git
```

Then, 
1. Please download the saved_models from here : [here](https://doi.org/10.6084/m9.figshare.26172565.v2)
2. paste the folder contents (models) in `Code/saved_models/` directory.

NOTE :
1. Model name : "complete_epoch_010_metric_0.95236.pth.tar" -> JUPPId C2(human) fold1 PAM120 model
2. Model name : "complete_epoch_010_metric_0.91918.pth.tar" -> sCerevisiae PAM120 model

Usage  :
```bash
  cd DensePPI-2/Code/
  conda activate denseppi_2
  cd JUPPId/Blosum45/                             #example, change directory according to need
  python3 main_train_test_blosum45JUPPId_C1.py    #example, change filename according to need
```

### Repository contents and directory structure 

To understand and replicate our work, the following is an overview of the scripts in different directories. The outputs and intermediates are supposed to be generated at distinct locations only, providing non-conflicting outputs for future use.

```
DensePPI-2
├── Code                    # Parent folder for all training and testing Python codes and .ipynb's
│   ├── input               # Contains input data for test i.e. sCerevisiae, human, SarsCov2
│   │   ├── JUPPId
│   │   ├── SarsCov2Data
│   │   └── sCerevisiaeData
│   ├── JUPPId              # Contains scripts to train and test on JUPPIdata(C1, C2, C3), SarsCov2 
│   │   ├── Blosum45
│   │   ├── Blosum90
│   │   ├── DensePPI
│   │   ├── PAM120
│   │   └── PAM250
│   ├── samplePlotsAndInterimResults      # Contains scripts for plot generation and prediction on random data
│   │   ├── test_models     # Contains scripts to generate prediction results with different strides
│   │   └── ...
│   ├── saved_models        # Contains best saved model for sCerevisiae PAM120 encoded data. 
│   ├── sCerevisiae         # Contains scripts to train and test on sCerevisiae
│   │   ├── Blosum45
│   │   ├── Blosum90
│   │   ├── PAM120
│   │   └── PAM250
│   ├── Utils               # Contains all the necessary utility scripts to run the main train/test scripts
│   └── model.py            # Contains classes and utils for the model used in this study 
├── sample_prediction_argParse_FASTALike  #Contains sample code base to test your input with PAM120-SC model.
├── environment.yml
└── README.md
```

### To run a sample prediction pipeline through arguments in the terminal

Please follow this tutorial to do a random test on sCerevisiae-PAM120 model on your chosen data in FASTALike format : [HERE](sample_prediction_argParse_FASTALike/README.md#running-our-prediction-in-terminal-linux)

#### NOTE :

All of the experiments were conducted on :
```
  2 x NVIDIA Quadro P5000 16GB
  NVIDIA Driver Version: 545.23.08    
  CUDA Version: 12.3 
  72 core - Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz
  512 GB RAM - 64GBx8 in paralleL
```
