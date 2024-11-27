from indellm import scorer
import argparse

def main(plm_name, tokenizer_name, csv_data, results_path, disable_tqm):

    # Initialize Scorer
    plm_scorer = scorer.Scorer()
    # Pick PLLM to use
    plm_scorer.initialize_plm(plm_name, tokenizer_name)
    # Score results
    plm_scorer.load_csv_to_score(csv_data, results_path)
    # Obtained data can be further processed here
    print("Scoring sequences")
    wt_perposition, mut_perposition, df_final = plm_scorer.score_data(masked=False, disable_tqm=disable_tqm)
    print("Scoring Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to score PLLR of indels.")

    parser.add_argument("--plm", type=str, 
                        help="Name of the PLM model: The following are accepted names: ESM2_8M, ESM2_35M, ESM2_150M, ESM2_650M, ESM2_3B, ESM2_15B, ESM1b, ESM1v, BertRost\n Alternatively give a path of a HuggingFace model")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer to use. This is only required for running a model not on the predefined list.")
    parser.add_argument("--csv_data", type=str, help="Location of the csv with the data to process.")
    parser.add_argument("--out_path", type=str, help="Path to use when storing the results of the scoring.")
    parser.add_argument("--disable_progres_bar", action="store_true", help="Hide progres bar.")

    # Parse the arguments
    args = parser.parse_args()

    # Pass the arguments to the main function
    main(args.plm, args.tokenizer, args.csv_data, args.out_path, args.disable_progres_bar)