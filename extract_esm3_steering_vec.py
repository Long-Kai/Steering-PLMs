import argparse
import pandas as pd
import torch
from utils.esm3_utils import load_esm3_model, get_esm3_layer_and_feature_dim, extract_esm3_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to the CSV file containing sequences and property score.")
    parser.add_argument('--theshold_pos', type=float, required=True, help="Threshold for positive data set.")
    parser.add_argument('--theshold_neg', type=float, required=True, help="Threshold for negative data set.")
    parser.add_argument('--property', type = str, required=True, help="Property to filter sequences by (e.g., 'therm', 'sol', 'GFP').")

    parser.add_argument('--num_data', type=int, default=None, help="Number of sequences to process. If None, processes all sequences.")
    parser.add_argument('--save_folder', type=str, default="saved_steering_vectors", help="Folder path to save steering vectors.")

    args = parser.parse_args()

    if args.theshold_pos <= args.theshold_neg:
        raise ValueError("Threshold for positive data must be greater than threshold for negative data.")

    df = pd.read_csv(args.data_path)
    pos_seqs = df['sequence'][df['score']>=args.theshold_pos].to_list()
    neg_seqs = df['sequence'][df['score']<=args.theshold_neg].to_list()
    
    if args.num_data is not None:
        pos_seqs = pos_seqs[:args.num_data]
        neg_seqs = neg_seqs[:args.num_data]

    model, tokenizer = load_esm3_model()
    n_layers, _ = get_esm3_layer_and_feature_dim() 

    pos_seq_repr_mat = extract_esm3_features(pos_seqs, model, tokenizer, n_layers)
    neg_seq_repr_mat = extract_esm3_features(neg_seqs, model, tokenizer, n_layers)

    pos_steering_vectors, neg_steering_vectors = [], []
    for i in range(n_layers):
        pos_steering_vectors.append(pos_seq_repr_mat[i].mean(dim=0))
        neg_steering_vectors.append(neg_seq_repr_mat[i].mean(dim=0))

    pos_steering_vectors = torch.stack(pos_steering_vectors).detach().cpu()
    neg_steering_vectors = torch.stack(neg_steering_vectors).detach().cpu()

    torch.save((pos_steering_vectors, neg_steering_vectors), f"{args.save_folder}/ESM3_{args.property}_steering_vectors.pt")