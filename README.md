# EnMAS-Encoding the Mutation rates of Amino acids for Sequence based PPI prediction

![Workflow](https://github.com/CMATERJU-BIOINFO/EnMAS/assets/56863228/25884b58-f86d-4bc8-aee7-3b8537a01445)



Identifying interactions between two or more proteins is crucial as it helps understand living organisms' cellular behavior and the underlying molecular mechanisms of various diseases. However, most existing computational algorithms in the field model this as binary interaction between any two proteins, instead of conserving the evolutionary regions of protein function and interactions. This is important for predicting potential interaction sites, vital for drug design, target identification, and understanding disease progression and pathogenic mechanisms. Position-aware encoding provides a way to incorporate the order of amino acids in a protein sequence into the model, thus capturing folding patterns, leading to more accurate predictions of protein structures and, consequently, their interactions. 
This is crucial because the sequence order can affect the structure and function of proteins.

The proposed model, EnMAS, is a novel bio-inspired substitution matrix-based sequence encoding with deep learning for identifying interacting protein pairs. It demonstrates an AUC of 97.13\% on the S. cerevisiae dataset, improving by 1.4\% over the best existing methods. Furthermore, EnMAS outperforms recent sequence-based approaches on the human benchmark dataset, addressing the complexities of protein-protein interaction test classes. EnMAS has been successfully applied for (a) identifying pathogen-host interactions (PHIs) (b) predicting near residue-level interaction even though the model was not trained on residue-level data. The enhanced performance on diverse test sets proves the efficiency of the bio-inspired sequence to image colour encoding strategy using the substitution matrices.



## Repository Contents and Data Directory

EnMAS-main\Code\Juppi
1. Data folder contains human sequences and csv files indicating ten fold distribution of classes C1, C2 and C3 
2. Remaining five sub-folders are there one for each method i.e. BLOSUM 45, BLOSUM 90, PAM 120, PAM 250 and DensePPI
3. Inside these five folders, main python codes can be found : main_Cx.py  x = [1,2,3]
4. Main file calls individual functions e.g. 
	a. Create folders if not exists
	b. Generates sub images for training and testing 
	c. Trains the deep learning model, and saves the model
	d. Predict test images and generate metrices

EnMAS-main\Code\yeast
1. Data folder contains input data i.e. yeast sequences for train and test.
2. Utils folder contains various functions e.g. data read, sub-image generation etc. 
3. Remaining four sub-folders are there one for each method i.e. BLOSUM 45, BLOSUM 90, PAM 120, PAM 250. 
Experimental results of DensePPI was taken from the paper itself. So no folder for DensePPI

