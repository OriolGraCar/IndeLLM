The Analysis folder includes an analysis notebook and subsetted datasets.

### Analysis.ipynd.
* All plots and analyses are presented in the paper and supplement material

### Indels_Cannonetal.csv: 3478 indels curated from Cannon et al. 
* id: Unique identifier for each indel
* reference: The protein or transcript ID 
* indel_type: insertion or deletion
* Classification: Benign or Pathogenic 
* Label: Binary label of classification, 1 is Pathogenic and 0 is Benign
* Subset: Cannon et al., split by Origin (DDD and ClinVar)
* wt_seq: wildtype sequence (cut if the native sequence is longer than 1022 amino acid)
* mut_seq: mutated sequence (cut if the native sequence is longer than 1022 amino acid)
* CADD_Cannonetal, CAPICE_Cannonetal, FATHMM_Cannonetal, MutPred_Cannonetal, MutationTaster2021_Cannonetal, Provean_Cannonetal, SIFT_Cannonetal, VEST_Cannonetal and VVP_Cannonetal: Predictions from each to extracted from the Cannon et al. dataset

### metrics.csv: 
* All calculated AUC, F1 and MCC scores for every dataset split, PLM, scoring approach and Siamese Model replicates.


### siamese_m4_(alldata/test/train/val).csv: 
* Scores from all replicates and mean from Siamese Model 4 for all ids (matching the ids in Data/Indels.csv). We have split the id into test, train and validation datasets. 

