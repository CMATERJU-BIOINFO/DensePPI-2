# EnMAS-Encoding the Mutation rates of Amino acids for Sequence based PPI prediction

[Workflow1.pdf](https://github.com/user-attachments/files/16071207/Workflow1.pdf)


EnMAS, an innovative substitution matrix-based deep learning approach for identifying interacting protein pairs. BLOcks SUbstitution Matrix (BLOSUM) and Point Accepted Mutation (PAM) matrices are usually used to score amino acid peptide sequence alignment between evolutionary divergent and closely related species. Contrary to the random colour generation approach in DensePPI, the mutation rate of amino acids in a protein sequence over long evolutionary time scales from these substitution matrices are used to generate 2D images from the amino acid sequences of the interacting proteins. A variation of the Convolution Neural Network, DenseNet201 is used for the training and classification of these images. The performance is evaluated on both benchmarked datasets _JUPPI_ and _Saccharomyces Cerevisiae_ (S.cerevisiae). The proposed model, EnMAS, demonstrates an AUC (Area Under the Curve) of 97.35\% when applied to the _S. cerevisiae_ dataset, marking an improvement of 1.57\% compared to the best existing methods. Furthermore, EnMAS outperforms the most recent sequence-based approaches on the JUPPI dataset, considering the complexities of the pair input method across distinct test classes of protein-protein interactions (PPIs). The enhanced performance on diverse test sets proves the efficiency of the bio-inspired sequence to image colour encoding strategy using the substitution matrices. 


## Citation

If you have used **EnMAS** in your research, please kindly cite the following publications:

("Coming soon!")
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

