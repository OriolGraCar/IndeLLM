from Bio.Align import PairwiseAligner

def get_indel_info(wt_seq, mut_seq):

    aligner = PairwiseAligner(score="blastp")
    aligner.mode = 'global'
    aligner.open_gap_score = -12
    aligner.extend_gap_score = 0

    alignment = aligner.align(wt_seq, mut_seq)[0]  # Take the best alignment
    if len(alignment.aligned[0]) > 2 or len(wt_seq) == len(mut_seq):
        print(alignment)
        raise Exception("Sequences must have only one indel")

    length_diff = abs(len(mut_seq) - len(wt_seq))
    if len(wt_seq) > len(mut_seq):
        indel_type = 0 # This means deletion
        if len(alignment.aligned[0]) > 1:
            start = alignment.aligned[0][0][1] # Location of the startindex of the first gap on the wt seq
            end = alignment.aligned[0][1][0] # Location of the end index of the first gap on the wt seq
        elif alignment.aligned[0][0][0] != 0: # This means the gap is in the first aminoacids
            start = 0
            end = alignment.aligned[0][0][0]
        else: # This means its on the last position
            start = alignment.aligned[0][0][1]
            end = len(wt_seq)
    else:
        indel_type = 1 # This means insertion
        if len(alignment.aligned[1]) > 1:
            start = alignment.aligned[1][0][1] # Location of the startindex of the first gap on the mut seq
            end = alignment.aligned[1][1][0] # Location of the end index of the first gap on the mut seq
        elif alignment.aligned[1][0][0] != 0: # This means the gap is in the first aminoacids
            start = 0
            end = alignment.aligned[1][0][0]
        else: # This means its in the last position
            start = alignment.aligned[1][0][1]
            end = len(mut_seq)

    if (end - start) != length_diff:
        print(alignment)
        raise Exception("Gap size does not match sequence length difference")

    return start, end, length_diff, indel_type


def truncate_sequences(wtseq, mutseq):


    start, end, length_diff, indel_type = get_indel_info(wtseq, mutseq)

    allowance = 500
    # compute max allowance for ESM1
    if length_diff > 22:
        extra = length_diff - 22
        extra = int(extra/2) + 1
        allowance = allowance - extra

    # adjust indexes to have them centered around 500 each side of the gap
    if indel_type == 0: # This means deletion
        start_w = start - allowance
        end_w = end + allowance
        start_m = start - allowance
        end_m = start + allowance
    else: # This means insertion
        start_m = start - allowance
        end_m = end + allowance
        start_w = start - allowance
        end_w = start + allowance
            
    wtseq = wtseq[start_w:end_w]
    mutseq = mutseq[start_m:end_m]

    return wtseq, mutseq

