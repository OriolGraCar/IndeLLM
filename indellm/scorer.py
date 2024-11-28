from indellm import berteval
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


class Scorer:
    def __init__(self):
        self.max_length = 1250 # extra 250 spaces for insertions
        self.df = None
        self.plm_model = None
        self.plm_name = ""
        self.results_path = None


    def load_csv_to_score(self, data_csv, results_path):
        self.df = pd.read_csv(data_csv)
        self.results_path = results_path
        if not os.path.exists(results_path):
            os.makedirs(results_path)


    def initialize_plm(self, model_name, tokenizer_name=None):
        self.plm_model = berteval.BertEval(model_name, tokenizer_name)
        name = os.path.basename(model_name).split(".")[0]
        self.plm_name = name


    def _compute_evofit(self, sequence, masked):
        self.plm_model.update_seq(sequence)
        if masked:
            results = self.plm_model.run_inference_masked()
        else:
            results = self.plm_model.run_inference()

        vals = np.zeros(self.max_length)
        for i,aa in enumerate(sequence):
            vals[i] = results[i][aa]
        return  vals
    
    @staticmethod
    def truncate_sequences(wtseq, mutseq):

        min_length = min(len(wtseq), len(mutseq))
    
        for i in range(min_length):
            if wtseq[i] != mutseq[i]:
                start_position = i # index of the first difference
        
        # Compute length
        wt_len = len(wtseq)
        mut_len = len(mutseq)
        start_i = 0
        end_i_wt = wt_len
        end_i_mut = mut_len
        diff = abs(wt_len - mut_len)
        allowance = 500
        if diff > 22: # ESM1 allowance
            extra = diff - 22
            extra = int(extra/2) + 1
            allowance = allowance - extra
        # recalculate index to have it centered at 500 each side
        if start_position > allowance:
            start_i = start_position - allowance
        if wt_len > mut_len:
            end_i_wt = start_position + allowance + diff
            end_i_mut = start_position + allowance
        else:
            end_i_wt = start_position + allowance
            end_i_mut = start_position + allowance + diff
        wtseq = wtseq[start_i:end_i_wt]
        mutseq = mutseq[start_i:end_i_mut]
        
        return wtseq, mutseq
    

    def score_data(self, masked=False, disable_tqm=True):

        if not self.df:
            raise("Load data first with .load_csv_to_score(data_csv, results_path)")
        
        # Create data holders for the new dataframes 
        wt_fit_results = {}
        mut_fit_results = {}
        score_dict = {"id":[], "Brandes_wt":[],"Brandes_mut":[], "indel_length":[], 
                      "IndeLLM_wt":[], "IndeLLM_mut":[], "IndeLLM_filtered":[], "label":[],
                      "wt_seq": [], "mut_seq":[]}


        for i, mutseq in enumerate(tqdm(self.df["mut_seq"], desc="Processing sequences", disable=disable_tqm)):
            wtseq = self.df["wt_seq"][i]
            s_id = self.df["id"]
            # Remove token X form sequence
            mutseq = mutseq.replace("X","")
            wtseq = wtseq.replace("X","")
            # Remove token * from sequence
            mutseq = mutseq.replace("*","")
            wtseq = wtseq.replace("*","")
            if "U" in wtseq or "U" in mutseq:
                print("Skipping sequence with unconventional aminoacid 'U'")
                continue

            # Check if the sequence is too long
            if (len(wtseq) > 1000 or len(mutseq) > 1000):
                # Find were the insertion or deletion happened
                #start_position = self.df["Protein_start"][i]
                wtseq, mutseq = self.truncate_sequences(wtseq, mutseq)

            # compute evofit for wt and mut
            wt_vals = self._compute_evofit(wtseq, masked=masked)
            mut_vals = self._compute_evofit(mutseq, masked=masked)

            # Compute scores
            gwt_score, gmut_score, localwt_score, localmut_score, clean_score = self.compute_PLLR(wtseq, mutseq, wt_vals, mut_vals) 

            # Extract information
            try:
                label = self.df["label"][i]
            except:
                label = -1

            score_dict["label"].append(label)
            score_dict["id"].append(s_id)
            score_dict["Brandes_wt"].append(gwt_score)
            score_dict["Brandes_mut"].append(gmut_score)
            score_dict["IndeLLM_wt"].append(localwt_score)
            score_dict["IndeLLM_mut"].append(localmut_score)
            score_dict["IndeLLM_filtered"].append(clean_score)
            score_dict["indel_length"].append(len(mutseq) - len(wtseq))
            score_dict["wt_seq"].append(wtseq)
            score_dict["mut_seq"].append(mutseq)

            wt_fit_results.setdefault("id", []).append(s_id)
            for j in range(len(wt_vals)):
                wt_fit_results.setdefault(str(j), []).append(wt_vals[j])
                mut_fit_results.setdefault(str(j), []).append(mut_vals[j])

        
        # Convert results to DataFrame and csv
        wt_fit_results = pd.DataFrame(wt_fit_results)
        mut_fit_results = pd.DataFrame(mut_fit_results)
        wt_fit_results.to_csv(os.path.join(self.results_path, f"wt_fitnesses_{self.plm_name}.csv"), index=False)
        mut_fit_results.to_csv(os.path.join(self.results_path, f"mut_fitnesses_{self.plm_name}.csv"), index=False)
        
        df_final = pd.DataFrame(score_dict)
        df_final.to_csv(os.path.join(self.results_path, f"scores_{self.plm_name}.csv"), index=False)

        return wt_fit_results, mut_fit_results, df_final

    @staticmethod
    def compute_PLLR(wtseq, mutseq, wt_p, mut_p):

        scorelocal_wt = []
        scorelocal_mut = []
        # compute size of either insertion or deletion
        diff_len = len(wtseq) - len(mutseq)
        extra = 0
        if diff_len > 0: # this means deletion
            for i in range(len(mutseq)):
                wt_i = i + extra
                if wtseq[wt_i] != mutseq[i]:
                    extra = diff_len
                    wt_i = i + extra
                    score_wt = wt_p[wt_i]
                    score_mut = mut_p[i]
                    scorelocal_wt.append(score_wt)
                    scorelocal_mut.append(score_mut)

        else: # this means insertion
            for i in range(len(wtseq)):
                mut_i = i + extra
                if wtseq[i] != mutseq[mut_i]:
                    extra = -diff_len
                    mut_i = i + extra
                    score_wt = wt_p[i]
                    score_mut = mut_p[mut_i]
                    scorelocal_wt.append(score_wt)
                    scorelocal_mut.append(score_mut)

        # Compute final scores
        global_wt = np.sum(wt_p)
        global_mut = np.sum(mut_p)
        scorelocal_wt = np.array(scorelocal_wt)
        scorelocal_mut = np.array(scorelocal_mut)
        local_wt = np.sum(scorelocal_wt)
        local_mut = np.sum(scorelocal_mut)

        localdiff = scorelocal_mut - scorelocal_wt
        f_score = 0
        for e in localdiff:
            if np.abs(e) > 0.07:
                f_score += e

        return global_wt, global_mut, local_wt, local_mut, f_score


