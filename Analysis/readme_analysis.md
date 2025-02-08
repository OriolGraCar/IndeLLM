The Analysis folder include 

Analysis.ipynd notebook - All plots and analysis prested in the article and supplement material

Indels_Cannonet al: 3478 indels curated from Cannon et al. 
    id: Unique identifier for each indel
    reference: The protein or transcript ID 
    indel_type: Insertion or deletion
    Classification: Benign or Pathogenic 
    Label: Binary lable of classification, 1 is Pathogenic and 0 is Benign
    Subset: Cannon et al, split by Origion (DDD and ClinVar)
    wt_seq: wildtype sequence (cut if native sequence is longer than 1022 amino acid)
    mut_seq: mutated sequence (cut if native sequence is longer than 1022 amino acid)
    CADD_Cannonetal, CAPICE_Cannonetal, FATHMM_Cannonetal, MutPred_Cannonetal, MutationTaster2021_Cannonetal, Provean_Cannonetal, SIFT_Cannonetal, VEST_Cannonetal and VVP_Cannonetal: Predictions from each tool, extracted from the Cannon et al dataset

Metrics.csv: All calcualted AUC, F1 and MCC scores for every dataset split, PLM, scoring approach and Siamese Model replicates


Siamese_m4_(alldata / test / train / val).csv: scores from all replicates and mean from Siamese Model 4, for all ids (matching the ids in Data/Indels.csv). We have split the id into test, train and validation datasets. 

