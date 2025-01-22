from indellm import model
import argparse
import os


def main(args):
    
    # Arguments to use
    model = args.model
    plm_name = args.plm
    embedding_path = args.embd_path
    output_path = args.out_path
    output_name = args.out_name
    csv_data =  args.csv_data
    disable_tqdm = args.disable_progres_bar
    threshold = args.threshold
    print(f"Running Inference using trained model: {model}")
    print(f"Using {plm_name} for the embedding. Be sure the model was trained to work with this particular Embedding")

    # Model initialization
    # Default arguments 
    # if model was trained with other arguments those need to be updated batch_size=32, learning_rate =0.01, hidden_layer_size = 8, n_slope = 0.1
    m = model.Indellm() 
    if disable_tqdm:
        m.disable_tqdm()
    m.run(csv_data, model, plm_name, embedding_path, output_path, output_name, threshold)
    
    print("Predictions Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to score PLLR of indels.")

    parser.add_argument("--model", type=str, required=True, help="Location of the trained model")
    parser.add_argument("--plm",  type=str, choices=["ESM2_8M", "ESM2_35M", "ESM2_150M", "ESM2_650M", "ESM2_3B", "ESM2_15B", 
                                                     "ESM1b", "ESM1v_1", "ESM1v_2", "ESM1v_3", "ESM1v_4", "ESM1v_5"],
                                                     default="ESM2_650M", help="PLM to use to extract embeddings")

    parser.add_argument("--csv_data", type=str, required=True, help="Location of the csv with the data to evaluate.")
    parser.add_argument("--out_path", type=str, required=True, help="Path to use when storing the results of the predictions.")
    parser.add_argument("--out_name", type=str, default ="Predictions", help="Name for the txt file to generate.")
    parser.add_argument("--embd_path", type=str, required=True, help="Path to use when storing the embeddings of the csv data.")
    parser.add_argument("--threshold", type=float, default=0.46, help="Threshold to use to label data.")
    parser.add_argument("--disable_progres_bar", action="store_true", help="Hide progres bar.")

    # Parse the arguments
    args = parser.parse_args()

    # Pass the arguments to the main function
    main(args)
