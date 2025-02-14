The Data folder include generated datasets in .csv format

### Indels.csv: 7500 indels curated from three publications. 
* id: Unique identifier for each indel
* reference: The protein or transcript ID 
* indel_type: Insertion or deletion
* Classification: Benign or Pathogenic 
* Label: Binary lable of classification, 1 is Pathogenic and 0 is Benign
* Subset: the 7500 indels are collected from Cannon et al, Fan et al, and Brandes et al. papers, which is indexed here. For Cannon et al, we have splitted by origin (DDD and ClinVar)
* wt_seq: wildtype sequence (cut if native sequence is longer than 1022 amino acid)
* mut_seq: mutated sequence (cut if native sequence is longer than 1022 amino acid)
* Provean: Predictions from Provean 
* MutPred-Indel: Predictions from MutPred-Indel


### scores_(ESM1b, ESM1v, ESM2_3b/150M/650M/650M_masked, ESM3, ProtBert).csv: zero-shot inference scores per PLM.
* id: Unique identifier for each indel, matching Indels.csv
* Brandes_wt: Brandes score for wt sequence
* Brandes_mut: Brandes score for mut sequence
* IndeLLM_wt: IndeLLM score for wt sequence
* IndeLLM_mut: IndeLLM score for mut sequence
* label: Binary lable of classification, 1 is Pathogenic and 0 is Benign
* indel_length: lenght of inserted (positive values) or deleted (negative values) amino acids
* indel_type: Insertion or deletion
* subset: Same as Indels.csv (notes if indel are collected from Cannon et al, Fan et al, or Brandes et al. papers)
* wt_seq: wildtype sequence (cut if native sequence is longer than 1022 amino acid)
* mut_seq: mutated sequence (cut if native sequence is longer than 1022 amino acid)


### train.csv: Training dataset contains 5960 indels (Indetical columns as in Indels.csv)
### val.csv: Validation dataset contains 819 indels (Indetical columns as in Indels.csv)
### test.csv: Test dataset contains 721 indels (Indetical columns as in Indels.csv)
    