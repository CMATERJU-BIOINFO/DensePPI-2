### Running our prediction in terminal-linux

Please install our environment using documentation from : [HERE](../README.md#setting-up-the-environment-for-enmas-)

The input data `{input}` should be a fasta file with entries like the following :
```bash
  Index        Protein_1_ID         protein_2_ID
  1 PA  PB
  >PA
  MLGNSAPGPATR...
  >PB
  MENFQKVEKIGE...
  2 PB  PC
  >PB
  MENFQKVEKIGE...
  >PC
  MSIMGRIKMSVN...
  3 PD  PE
  >PD
  MSRPQGLLWLPL...
  >PE
  MYQRMLRCGAEL...
```

To run it on terminal do :
```bash
  cd ../EnMAS/sample_prediction_argParse_FASTALike/
  conda activate enmasppi
  python3 enmasPred.py --input sample_FASTALike.txt --output ResultFiles/ --stride 64 --interaction_type "positive" --device cpu
```

The result will be generated in the subfolder `{output}` with name :
```bash
  {interaction_type}_{sc_result_sc_pam120_stride}_{stride}.csv
```

The output is a CSV with the following headers :
```bash
    'Index' : Index of the prediction
    'Protein_1' : Name of interacting protein 1
    'Protein_2' : Name of interacting protein 2
    'Protein_seq1' : Sequence of interacting protein 1
    'Protein_seq2' : Sequence of interacting protein 2
    'img_dim' : Dimension for final matrix at protein level using strides (num of sub images),
    'resultant_scores' : Raw redictions for final matrix at protein level using strides (num of sub images)
    'orig_pred' : Final protein level prediction : 0 -> negative ; 1 -> positive
```