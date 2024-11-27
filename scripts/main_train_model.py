from indellm import model
import argparse
import os


def main(args):

    # Arguments to use
    model_name = args.model_name
    plm_name = args.plm
    embedding_path = args.embd_path
    model_path = args.model_path 
    output_path = args.out_path
    train_csv_data =  args.train_data
    test_csv_data = args.test_data
    val_csv_data = args.val_data
    disable_tqdm = args.disable_progres_bar


    # Initialize the Data handler to generate embeddings of the selected sequences using the selected PLM
    print("Generating Embeddings")
    data_hanlder = model.DataProcessor(plm_name, embedding_path)
    if disable_tqdm:
        data_hanlder.disable_tqdm()
    # Compute and save embeddings
    data_hanlder.read_csv(train_csv_data)
    data_hanlder.extract_embeddings()
    data_hanlder.read_csv(test_csv_data)
    data_hanlder.extract_embeddings()
    data_hanlder.read_csv(val_csv_data)
    data_hanlder.extract_embeddings()
    

    # Embedding file name will be equal to the csv file name
    train_name = os.path.basename(train_csv_data).split(".")[0]
    test_name = os.path.basename(test_csv_data).split(".")[0]
    val_name = os.path.basename(val_csv_data).split(".")[0]
    train_embd = os.path.join(embedding_path, train_name)
    test_embd = os.path.join(embedding_path, test_name)
    val_embd = os.path.join(embedding_path, val_name)

    # Model initialization
    print("Preparing Training")
    m = model.Indellm() #  default arguments batch_size = 32, learning_rate = 1e-4, hidden_layer_size = 1028
    if disable_tqdm:
        m.disable_tqdm()
    m.load_train_data(train_data=train_embd, test_data=test_embd, val_data=val_embd)
    # Train model
    m.train(model_path, model_name, n_epochs=100, early_stop=10)
    # Asses performance of final model on held out dataset
    m.assess_performance(output_path)
    
    print("Training Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to score PLLR of indels.")

    parser.add_argument("--model_name", type=str, required=True, help="Name for the model to train")
    parser.add_argument("--plm",  type=str, choices=["ESM2_8M", "ESM2_35M", "ESM2_150M", "ESM2_650M", "ESM2_3B", "ESM2_15B", 
                                                     "ESM1b", "ESM1v_1", "ESM1v_2", "ESM1v_3", "ESM1v_4", "ESM1v_5"],
                                                     default="ESM2_650M", help="PLM to use to extract embeddings")

    parser.add_argument("--train_data", type=str, required=True, help="Location of the csv with the training data.")
    parser.add_argument("--test_data", type=str, required=True, help="Location of the csv with the test data.")
    parser.add_argument("--val_data", type=str, required=True, help="Location of the csv with the validation data.")
    parser.add_argument("--out_path", type=str, required=True, help="Path to use when storing the results of the predictions.")
    parser.add_argument("--embd_path", type=str, required=True, help="Path to use when storing the embeddings of the csv data.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to use when storing the trainned models.")
    parser.add_argument("--disable_progres_bar", action="store_true", help="Hide progres bar.")

    # Parse the arguments
    args = parser.parse_args()

    # Pass the arguments to the main function
    main(args)