import torch
import esm
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.nn as nn
from tqdm import tqdm
import os 
import pandas as pd
import math
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from indellm import scorer
from indellm import utils

def set_random_seeds(seed: int):
    """
    Set all possible random seeds for reproducibility.
    :param seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using CUDA, set CUDA-specific seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For all GPUs


class Datahandler(Dataset):
    def __init__(self,row, datatype):
        super().__init__()
        self.seq = row[f'{datatype}_seq']
        self.label = row['label']
        self.unique_id = row['id']
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, idx):
        return (self.label[idx], self.seq[idx], self.unique_id)
    
class ModelData(Dataset):
    def __init__(self, X, y):
        super().__init__()

        self.seq = torch.tensor(X)
        self.label = torch.tensor(y)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        return self.seq[index], self.label[index]

class DataProcessor:
    def __init__(self, model_name, embedding_path, seed=42):
        self.model_name = model_name
        self.data = None
        self.model = None
        self.alphabet = None
        self.device = None
        self.seed = seed
        self.embedding_path = embedding_path
        self.embedding_batch_size = 32
        self.last_hlayer = 33
        self.data_name = ""
        self.tqdm_status = False
        set_random_seeds(self.seed)
        self._init_device()
        self._load_esm_model(self.model_name)

    def set_data_name(self, name):
        self.data_name = name

    def set_data(self, data):
        self.data = data

    def disable_tqdm(self):
        self.tqdm_status = True

    
    def read_csv(self, csv):
        self.data = pd.read_csv(csv)
        self.csv = csv
        self.data_name = os.path.basename(csv).split(".")[0]
        # Apply the function row by row
        truncated_seqs = self.data.apply(lambda row: utils.truncate_sequences(row["wt_seq"], row["mut_seq"]), axis=1)
        
        # Extract wt_seqs and mut_seqs from the tuples
        wt_seqs = truncated_seqs.apply(lambda x: x[0])
        mut_seqs = truncated_seqs.apply(lambda x: x[1])
        
        self.data["wt_seq"] = wt_seqs
        self.data["mut_seq"] = mut_seqs


    def _init_device(self):
        """
        Detect the available device (CUDA, MPS, or CPU) and print its details.
        :return: The detected device.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"There are {torch.cuda.device_count()} GPU(s) available.")
            print("Device name:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print("No GPU or MPS available, using the CPU instead.")


    def _load_esm_model(self, model_name):
        """
        Load the specified ESM model based on the input model name.
        :param model_name: Name of the ESM model (e.g., 'ESM2_650M').
        :return: The model and alphabet.
        """
        # Model names to actual model identifiers
        model_mapping = {
            # ESM2 Models
            "ESM2_8M": "esm2_t6_8M_UR50D",
            "ESM2_35M": "esm2_t12_35M_UR50D",
            "ESM2_150M": "esm2_t30_150M_UR50D",
            "ESM2_650M": "esm2_t33_650M_UR50D",
            "ESM2_3B": "esm2_t36_3B_UR50D",
            "ESM2_15B": "esm2_t48_15B_UR50D",
            
            # ESM1b Model
            "ESM1b": "esm1b_t33_650M_UR50S",
            
            # ESM1v Models
            "ESM1v_1": "esm1v_t33_650M_UR90S_1",
            "ESM1v_2": "esm1v_t33_650M_UR90S_2",
            "ESM1v_3": "esm1v_t33_650M_UR90S_3",
            "ESM1v_4": "esm1v_t33_650M_UR90S_4",
            "ESM1v_5": "esm1v_t33_650M_UR90S_5",
        }

        if model_name not in model_mapping:
            raise ValueError(f"Model {model_name} is not recognized. Available models: {list(model_mapping.keys())}")

        model_identifier = model_mapping[model_name]
        print(f"Loading model: {model_identifier}...")

        # Load the model and alphabet
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_identifier)
        self.last_hlayer = int(model_identifier.split("_")[1][1:])
        self.model.eval()  # Set model to evaluation mode
 
    @staticmethod
    def _collate_fn(batch):
        labels, sequences, unique_id = zip(*batch)
        return list(zip(labels, sequences)), unique_id
   
    def extract_embeddings(self):


        batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)

        wt_dataset = Datahandler(self.data, "wt")
        wt_dataloader = DataLoader(wt_dataset, batch_size=self.embedding_batch_size, collate_fn=self._collate_fn, shuffle=False, drop_last=False)
        mt_dataset = Datahandler(self.data, "mut")
        mt_dataloader = DataLoader(mt_dataset, batch_size=self.embedding_batch_size, collate_fn=self._collate_fn, shuffle=False, drop_last=False)
        concat = []
        indel_lens = []
        indel_types = []
        s = scorer.Scorer()
        for wt_tuple,mut_tuple in tqdm(zip(wt_dataloader,mt_dataloader),total=len(wt_dataloader), disable=self.tqdm_status):
            batch_labels, wt_seqs, wt_batch_tokens = batch_converter(wt_tuple[0])
            _, mut_seqs, mut_batch_tokens = batch_converter(mut_tuple[0])
                    
            wt_batch_lens = (wt_batch_tokens != self.alphabet.padding_idx).sum(1)
            mut_batch_lens = (mut_batch_tokens != self.alphabet.padding_idx).sum(1)
            with torch.no_grad():
                # Process batch
                wt_result = self.model(wt_batch_tokens.to(self.device), repr_layers=[self.last_hlayer])
                mut_result = self.model(mut_batch_tokens.to(self.device), repr_layers=[self.last_hlayer]) 

            wt_logits = wt_result['logits']  # batch_size, max_seq_len, tokens
            mut_logits = mut_result['logits'] # batch_size, max_seq_len, tokens
            wt_reps = wt_result["representations"][self.last_hlayer]
            mut_reps = mut_result["representations"][self.last_hlayer]
            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            wt_sequence_representations = []
            mut_sequence_representations = []
            indel_sequence_representations = []


            for i, tokens_len in enumerate(wt_batch_lens):

                wt_seq = wt_seqs[i]
                mut_seq = mut_seqs[i]
      
                start, end, length_diff, indel_type = utils.get_indel_info(wt_seq, mut_seq)

                if indel_type == 0: # This means deletion

                    # Now we found the indexes, slice the sequence
                    wt_sliced = torch.cat((wt_reps[i, 1 : start], wt_reps[i, end: wt_batch_lens[i] - 1]))
                    indel_sequence_representations.append(wt_reps[i, start: end].mean(0))
                    wt_sequence_representations.append(wt_sliced.mean(0))
                    mut_sequence_representations.append(mut_reps[i, 1 : mut_batch_lens[i] - 1].mean(0))


                else: # This means insertion
                        
                    # Now we found the indexes, slice the sequence
                    mut_sliced = torch.cat((mut_reps[i, 1 : start], mut_reps[i, end: mut_batch_lens[i] - 1]))
                    indel_sequence_representations.append(mut_reps[i, start: end].mean(0))
                    mut_sequence_representations.append(mut_sliced.mean(0))
                    wt_sequence_representations.append(wt_reps[i, 1 : wt_batch_lens[i] - 1].mean(0))
            
                indel_lens.append(length_diff)
                indel_types.append(indel_type)
            """
            # Convert logits to probabilities using softmax
            wt_probs = torch.nn.functional.softmax(wt_logits, dim=-1)  # Shape: (batch_size, seq_length, vocab_size)
            mut_probs = torch.nn.functional.softmax(mut_logits, dim=-1)  # Shape: (batch_size, seq_length, vocab_size)

            
            # Create the dictionary for amino acid probabilities

            for batch_idx, wt_sequence in enumerate(wt_seqs):
                mut_sequence = mut_seqs[batch_idx]
                wt_sequence_probs = []
                mut_sequence_probs = []

                # Iterate over wildtype
                for position, token in enumerate(wt_sequence):
                    token_idx = wt_batch_tokens[batch_idx, position + 1].item() # Get the token index
                    aa_probability = wt_probs[batch_idx, position + 1, token_idx].item()  # Get probability for the specific amino acid
                    wt_sequence_probs.append(aa_probability)  # Add to the list

                # iterate over mutant
                for position, token in enumerate(mut_sequence):
                    token_idx = mut_batch_tokens[batch_idx, position + 1].item() 
                    aa_probability = mut_probs[batch_idx, position + 1, token_idx].item()  
                    mut_sequence_probs.append(aa_probability)  

                # Compute PLM score
                _, _, local_wt, local_mut, _ = s.compute_PLLR(wtseq=wt_sequence, mutseq=mut_sequence, wt_p=wt_sequence_probs, mut_p=mut_sequence_probs)
                #plm_scores.append(local_mut - local_wt + 1)
                plm_scores.append(0)
                
            """
            # Stack and concat
            stack_wt = torch.stack(wt_sequence_representations)
            stack_mut = torch.stack(mut_sequence_representations)
            stack_indels = torch.stack(indel_sequence_representations)
            concat.append(torch.cat((stack_wt, stack_mut, stack_indels), dim=1))
        # Concat all batches and save    
        concat = torch.cat(concat, dim=0).detach().cpu() 
        final_result = {'x': concat, 'label': self.data["label"], 'lengths':indel_lens, "type":indel_types, 'id': self.data["id"]}
        
        if not os.path.isdir(f'{self.embedding_path}'):
            os.makedirs(f'{self.embedding_path}')  # create the dir for embeddings

        final_location = os.path.join(self.embedding_path, f"{self.data_name}_embedding.pt")
        
        torch.save(final_result,final_location) 
        # print(f"Embedding Saved at:{self.embedding_path}\n")


class MLPClassifier_LeakyReLu(nn.Module):

    def __init__(self, num_input, num_hidden, num_output, negative_slope):
        super(MLPClassifier_LeakyReLu, self).__init__()

        # Instantiate an one-layer feed-forward classifier
        self.hidden = nn.Linear(num_input, num_hidden)
        self.predict = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True, negative_slope=negative_slope),
            nn.Linear(num_hidden, num_output)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.predict(x)
        x = self.softmax(x)

        return x

class Indellm:

    def __init__(self, batch_size=32, learning_rate =0.01, hidden_layer_size = 8, n_slope = 0.1):
        # visual
        self.tqdm_status = False
        # data holders
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_valid = None
        self.y_valid = None
        self.data_df = None
        self.n_slope = n_slope
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.test_seq_ids = None
        # Model and parameter holders
        self.n_slope = n_slope
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.model = None
        self.device = None
        self._init_device()

    def disable_tqdm(self):
        self.tqdm_status = True

    @staticmethod
    def unpickler(dataset_path):

        pt_embeds = torch.load(dataset_path)
        data_X = np.array(pt_embeds['x'])
        # Compute max and min
        # rows_with_nan = np.where(np.isnan(data_X).any(axis=1))[0]

        # print("Row indices with NaN values:", rows_with_nan)
        lens = np.array(pt_embeds['lengths']).reshape(-1, 1)
        indel_type = np.array(pt_embeds['type']).reshape(-1, 1)
        data_y = pt_embeds['label']
        unique_id = pt_embeds['id']

        data_X = np.hstack((data_X, lens))
        data_X = np.hstack((data_X,indel_type))

        return data_X, data_y, unique_id
    

    def _init_device(self):
        """
        Detect the available device (CUDA, MPS, or CPU) and print its details.
        :return: The detected device.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"There are {torch.cuda.device_count()} GPU(s) available.")
            print("Device name:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print("No GPU or MPS available, using the CPU instead.")
    
   
    def _load_embedding_data(self, pickled_data, shuffle=False):
        x, y, s_id = self.unpickler(pickled_data)
        dataset = ModelData(x,y)
        d_loader = DataLoader(dataset, self.batch_size, shuffle)
        return x, y, s_id, d_loader
  

    def load_train_data(self, train_data, test_data, val_data):
        self.x_train, self.y_train, self.train_seq_ids, self.train_loader = self._load_embedding_data(train_data, shuffle=True)
        self.x_test, self.y_test, self.test_seq_ids, self.test_loader = self._load_embedding_data(test_data)
        self.x_valid, self.y_valid, self.val_seq_ids, self.val_loader = self._load_embedding_data(val_data)


    @staticmethod
    def _flat_accuracy(preds, labels):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    def train(self, storage_path, model_name, n_epochs, early_stop):

        model_size = self.x_train.shape[1]
        print('\n\n\n\n')
        print('=============== Model training start ===============')
  
        print('model_size: ', model_size)
        print('num_hidden: ', self.hidden_layer_size)
        
        self.model = MLPClassifier_LeakyReLu(num_input = model_size, num_hidden = self.hidden_layer_size, num_output = 2, negative_slope = self.n_slope).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        
        # Define the loss function
        criterion = nn.BCELoss(reduction='sum') 

        # Optimization algorithm.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                         num_warmup_steps= 0,
        #                                         num_training_steps= len(train_loader)*n_epochs)

        n_epochs, best_perf, step, early_stop_count = n_epochs, 0, 0, early_stop

        for epoch in range(n_epochs):
            self.model.train()  # Set the model to train mode.
            loss_record = []

            train_pbar = tqdm(self.train_loader, position=0, leave=True, disable=self.tqdm_status)

            for batch in train_pbar:
                optimizer.zero_grad()               # Set gradient to zero.
                # Move the data to device.
                b_seq, b_labels = tuple(t.to(self.device) for t in batch)
                pred = self.model(b_seq.float())
                b_labels = b_labels.float()
                loss = criterion(pred[:, 0], b_labels)

                # Compute gradient(backpropagation).
                loss.backward()
                optimizer.step()                    # Update parameters.
                # scheduler.step()

                step += 1
                loss_record.append(loss.detach().item())
                # Display current epoch number and loss on tqdm progress bar.
                train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
                train_pbar.set_postfix({'loss': loss.detach().item()})
            
            mean_train_loss = sum(loss_record)/len(loss_record)

            ########### =========================== Evaluation=========================################
            print('\n\n###########=========================== Evaluating=========================################\n\n')

            self.model.eval()  # Set the model to evaluation mode.
            loss_record = []

            preds_record = []
            labels_record = []

            val_pbar = tqdm(self.val_loader, position=0, leave=True, disable=self.tqdm_status)
            for batch in val_pbar:

                # Move your data to device.
                b_seq, b_labels = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    b_labels = b_labels.float()
                    pred = self.model(b_seq.float())
                    loss = criterion(pred[:, 0], b_labels)

                    # preds.append(pred[:,0].detach().cpu()[0].tolist())
                    # labels.append(b_labels.detach().cpu()[0].tolist())

                loss_record.append(loss.item())
                #total_eval_accuracy += self._flat_accuracy(pred, b_labels)

                val_pbar.set_description(f'Evaluating [{epoch + 1}/{n_epochs}]')
                val_pbar.set_postfix({'evaluate loss': loss.detach().item()})
                preds_record.append(pred[:,0].cpu().numpy())
                labels_record.append(b_labels.cpu().numpy())

            # For selecting the best MCC threshold
            # breakpoint()
            # y_true_np = np.array(labels)
            # pred_np = np.array(preds)
            # for label, pred_value in zip(y_true_np, pred_np):
            #     with open(f'./threhold_pick.txt', 'a+') as f:
            #         f.write(f'{label}\t{pred_value}\n')

            mean_valid_loss = sum(loss_record)/len(loss_record)
            #avg_val_accuracy = total_eval_accuracy / len(self.val_loader)
            labels_record =  np.concatenate(labels_record, axis=0)
            preds_record  =  np.concatenate(preds_record, axis=0)
            auc_value = roc_auc_score(labels_record, preds_record)

            print(f'\nEpoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}, AUC {auc_value:.4f}')

            # Save best models
            if auc_value > best_perf:
                best_perf = auc_value

                if not os.path.isdir(f'{storage_path}'):
                    # Create directory of saving models.
                    os.mkdir(f'{storage_path}')

                torch.save({
                    'model_state_dict': self.model.state_dict(), },
                    f'{storage_path}/{model_name}.ckpt')  # Save the best model

                print('\nSaving model with loss {:.3f}...'.format(mean_valid_loss))
                print(f"Epoch {epoch}")

                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= early_stop:
                print('\nModel is not improving, halting train session.')
                return

    def assess_performance(self, model_path, model_name, output_path):
  
        print('=============== Predicting & Evaluating the trained model ===============')
        checkpoint=torch.load(f'{model_path}/{model_name}.ckpt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        results_val = self._predict(self.val_loader, self.val_seq_ids, train=True, output_path=output_path)
        results_train = self._predict(self.train_loader, self.train_seq_ids, train=True, output_path=output_path)
        results_test = self._predict(self.test_loader, self.test_seq_ids, train=True, output_path=output_path)
        return results_train, results_val, results_test

    
    def _predict(self, data_loader, record_id, train=False, output_path=None, output_name=None,threshold=0.46):

        self.model.eval()  # Set the model to evaluation mode.
        preds = []
        labels = []
        for batch in tqdm(data_loader, disable=self.tqdm_status):
                b_seq, b_labels = tuple(t.to(self.device) for t in batch)
                with torch.no_grad():
                    pred = self.model(b_seq.float())
                    preds.append(pred[:, 0].detach().cpu())
                    labels.append(b_labels.detach().cpu())
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        if not os.path.exists(f'{output_path}'):
                os.makedirs(output_path)

        if train:

            label_names = {'0': 0, '1': 1}
            auc_value = roc_auc_score(labels, preds)
            print('AUC score: ', auc_value)

            return auc_value

            # Saving the prediction results for each test data
            #if not os.path.exists(f'{output_path}/model_eval_result.txt'):
            #    header = "target_id\tlabel\tprediction\n"
            #    with open(f'{output_path}/model_eval_result.txt', 'a') as file_writer:
            #        file_writer.write(header)
    
            #for ids, label, pred_value in zip(record_id, y_true_np, preds):
            #    with open(f'{output_path}/model_eval_result.txt', 'a+') as f:
            #        f.write(f'{ids}\t{label}\t{pred_value}\n')

            #with open(f'{output_path}/model_performance.txt', 'a') as file_writer:
            #    file_writer.write(f'MCC: {MCC}\nroc_auc_score: {auc_value}\n')

        else:

            preds2 = np.array(preds >= threshold, dtype=int)

            header = "id,score,prediction\n"

            with open(f'{output_path}/{output_name}.txt', 'w') as file_writer:
                file_writer.write(header)

            for ids, pred_value, pred_label in zip(record_id, preds, preds2):
                with open(f'{output_path}/{output_name}.txt', 'a+') as f:
                    f.write(f'{ids},{pred_value},{pred_label}\n')


    def run(self, data_location, model_location, plm_name, embedding_path, output_path, output_name, threshold=0.46):
        '''
        '''
        # Check type of data
        if isinstance(data_location, str):
            target_df = pd.read_csv(data_location)
            print(f'Generating embedings for {data_location}')
            data_name = os.path.basename(data_location).split(".")[0]
        else: # This means is a DataFrame already
            target_df = data_location
            print("Generating embedings")
            data_name = plm_name
            
        target_df['label'] = -1 # To fill column
        self.data_df = target_df
        d = DataProcessor(plm_name, embedding_path, 42) 
        d.set_data(target_df)
        d.set_data_name(data_name)
        d.extract_embeddings()
        embedding_location = os.path.join(embedding_path, f"{data_name}_embedding.pt")

        x_target, y_target, record_id = self.unpickler(embedding_location)
        #print('X_target shape: ', x_target.shape)

        target_dataset = ModelData(x_target, y_target)
        target_loader = DataLoader(target_dataset, batch_size=32)

        model_size = x_target.shape[1]
        self.model = MLPClassifier_LeakyReLu(num_input = model_size, num_hidden = self.hidden_layer_size, num_output = 2, negative_slope = self.n_slope).to(self.device)

        if not os.path.exists(model_location):
            print('No model found')

        checkpoint=torch.load(model_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self._predict(target_loader, record_id, train=False, output_path=output_path, output_name=output_name, threshold=threshold)


        print()
        print(f"Your prediction results are saved in {output_path}/{output_name}.txt")

