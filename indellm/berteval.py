from typing import Any
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer, AutoModelForMaskedLM
import transformers
import torch
transformers.logging.set_verbosity_error()

class bertModels:
  def __init__(self, topK, modelName, tokenizer_name=None):
    self.topK = topK
    self.modelName = modelName
    self.tokenizer_name = tokenizer_name

    # Rostlab model 460M parameters
    if modelName == 'BertRost':
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")

    # ESM-1b with 650M parameters trained on UniRef50
    elif modelName == "ESM1b":
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S")

    # ESM-1 with 650M parameters trained on UniRef90
    elif modelName == "ESM1v":
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")
        self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm1v_t33_650M_UR90S_1")

    # ESM-2 with 650M parameters trained on UR50D
    elif modelName == "ESM2_650M":
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # ESM-2 with 3b parameters
    elif modelName == "ESM2_3B":
        # add check RAM
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
        self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")

    # ESM-2 with 150M parameters
    elif modelName == "ESM2_150M":
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t30_150M_UR50D")

    # ESM-2 with 35M parameters
    elif modelName == "ESM2_35M":
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")

    # ESM-2 with 15b parameters
    elif modelName == "ESM2_15B":
        # add check RAM
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
        self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t48_15B_UR50D")


    # Any other model just fetch it
    else:
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.modelName)

    self.unmasker = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, top_k = self.topK)

class BertEval(bertModels):
    def __init__(self, modelName, tokenizername=None, topK=25):
       super(BertEval, self).__init__(topK, modelName, tokenizername)
       self.seq = ""
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.model.to(self.device)
       self.model.eval()

    def update_seq(self, seq):
       self.seq = seq

    def str_prepare(self, mut_indx, mask="<mask>"):
        list_seq = list(self.seq)
        self.seq = " ".join(list_seq)
        if self.modelName == "BertRost":
            self.seq = self.seq.replace("X", "[MASK]")
        elif "ESM" in self.modelName:
            self.seq = self.seq.replace("X", "<mask>")
        else:
            if mask:
                self.seq = self.seq.replace("X", mask)
            else:
                print("For model %s a mask must be specified" % self.modelName)
                exit(1)

    def find_positions(self):
        mut_indx = []
        for i,ch in enumerate(self.seq):
            if ch == "X":
                mut_indx.append(i)
        return mut_indx

    def itter_dict(self, dct, idx):
        new_dct = {}
        new_dct[idx] = {}
        for i in range (len(dct)):
            new_dct[idx][dct[i].get("token_str")] = dct[i].get("score")
        return  new_dct

    def run(self):
        # find positions to eval
        mut_indx = self.find_positions()
        # change name of masking
        self.str_prepare(mut_indx)
        # eval
        ans = self.unmasker(self.seq)
        dict_all = {}
        for i in range(len(mut_indx)):
            try:
                new_dct = self.itter_dict(ans[i], mut_indx[i])
                dict_all.update(new_dct)
            except: # Single Mutation
                new_dct = self.itter_dict(ans, mut_indx[i])
                dict_all.update(new_dct)
                break

        return dict_all

    def run_inference_masked(self):
        # Run all positions
        # Tokenize the input sequence
        input_ids = self.tokenizer.encode(self.seq, return_tensors="pt").to(self.device)
        sequence_length = input_ids.shape[1] - 2  # Excluding the special tokens
        vocab_order = self.tokenizer.get_vocab()

        # List of amino acids
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        dict_all = {}

        # Calculate LLRs for each position and amino acid
        for position in range(sequence_length):
            p = position + 1
            # Mask the target position
            masked_input_ids = input_ids.clone()
            masked_input_ids[0, p] = self.tokenizer.mask_token_id

            # Get logits for the masked token
            with torch.no_grad():
                logits = self.model(masked_input_ids).logits

            # Calculate  probabilities
            probabilities = torch.nn.functional.softmax(logits[0, p], dim=0)
            aa_pos = {}
            for i,aa in enumerate(amino_acids):
                aa_pos[aa] = float(probabilities[vocab_order[aa]])
            dict_all[position] = aa_pos

        return dict_all

    def run_inference(self):
        # Run all positions
        # Tokenize the input sequence
        input_ids = self.tokenizer.encode(self.seq, return_tensors="pt").to(self.device)
        sequence_length = input_ids.shape[1] - 2  # Excluding the special tokens
        vocab_order = self.tokenizer.get_vocab()

        # List of amino acids
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        dict_all = {}
        #input_ids[0, 1] = self.tokenizer.mask_token_id
        with torch.no_grad():
            logits = self.model(input_ids).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        for position in range(sequence_length):
            p = position + 1
            aa_pos = {}
            for i,aa in enumerate(amino_acids):
                aa_pos[aa] = float(probabilities[0][p][vocab_order[aa]])
            dict_all[position] = aa_pos

        return dict_all