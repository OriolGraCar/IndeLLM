import pandas as pd

# Import csv
indels = pd.read_csv("../Data/Indel_curated.csv")
indels_df = pd.DataFrame(indels)

# Change "-" to empty string
indels_df['wt'] = indels_df['wt'].replace('-', '')
indels_df['mut'] = indels_df['mut'].replace('-', '')

print(len(indels_df.index))

# Check if there is * or ? in the data
filter = indels_df['wt'].str.contains(r'\*|\?', regex=True, na=False) 
filter2 = indels_df['mut'].str.contains(r'\*|\?', regex=True, na=False)

# Drops rows with * or ?
indels_df = indels_df[~filter]
indels_df = indels_df[~filter2]

print(len(indels_df.index)) #removed 3 rows

#make mutated sequences
def mutate(sequence, protein_start, wt, mut):
    protein_start = int(protein_start) - 1
    wt = str(wt)
    mut = str(mut)
    wt_len = len(wt)
    sequence_before_wt = sequence[:protein_start]
    sequence_after_wt = sequence[protein_start + wt_len:]
    return sequence_before_wt + mut + sequence_after_wt

# Apply the function to create mutated sequences
indels_df['mutseq'] = indels_df.apply(lambda row: mutate(row['seq'], row['Protein_start'], row['wt'], row['mut']), axis=1)

# Add information on indel type and length
indels_df['seq'] = indels_df['seq'].astype(str)
indels_df['mutseq'] = indels_df['mutseq'].astype(str)
indels_df['length_wt'] = indels_df['seq'].apply(len)
indels_df['length_mut'] = indels_df['mutseq'].apply(len)
indels_df['length_indel'] = indels_df.apply(lambda row: abs(row['length_mut'] - row['length_wt']), axis=1)
indels_df['indel_type'] = indels_df.apply(lambda row: 'insertion' if row['length_mut'] > row['length_wt'] else 'deletion', axis=1)

# Check  indel type match the consequence
indels_df['check'] = indels_df.apply(lambda row: row['indel_type'] in row['Consequence_x'], axis=1)
print(indels_df['check'].value_counts())

# Check that I dont have any 0 indels
print(indels_df[indels_df['length_indel'] == 0])

indels_df.to_csv("Indel_curated_newmutseq.csv", sep=',')