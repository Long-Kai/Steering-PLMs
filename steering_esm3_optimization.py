import pandas as pd
import argparse
import numpy as np
from itertools import islice
from tqdm import tqdm
import types
import torch
import torch.nn.functional as F

from esm3.utils.constants.esm3 import SEQUENCE_MASK_TOKEN
from esm3.utils.decoding import decode_sequence

from module.steerable_esm3 import steering_forward, esm3_steering_forward
from utils.esm3_utils import load_esm3_model, pred_tokens, get_tokenwise_representations
from utils.opt_utils import topk_intersection_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =========================== must be provided ===============================
    parser.add_argument('--data_path', type = str, required=True, help="path to data file containing the sequences for optimization")

    #=========================== default is set ======================================
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")

    parser.add_argument('--property', type = str, default = "therm", help="therm, sol")
    parser.add_argument('--output_file', type = str, default=None, help="path to save the optimized sequences")

    parser.add_argument('--alpha', type = float, default = 1.0)
    parser.add_argument('--sv_from', type=str, default="saved_steering_vectors", help="folder path to load steering vectors from")

    parser.add_argument('--n', type = int, default = 1000, help="maximal number of sequences to optimize")
    parser.add_argument('--round', type = int, default = 1, help="number of optimization rounds")
    parser.add_argument('--T', type = int, default = 1, help="number of mutation sites per round")
    #============================================================================
    args = parser.parse_args() 

    # load reference data
    df = pd.read_csv(args.data_path)
    org_seqs = df['sequence'].to_list()
    
    # load model
    model, tokenizer = load_esm3_model()
    model.transformer.steering_forward = types.MethodType(steering_forward,  model.transformer)
    model.steering_forward = types.MethodType(esm3_steering_forward, model)
    
    # load steering vector
    steering_vectors_path = f"{args.sv_from}/ESM3_{args.property}_steering_vectors.pt"
    pos_steering_vectors, neg_steering_vectors = torch.load(steering_vectors_path)
    steering_vectors = pos_steering_vectors - neg_steering_vectors
    steering_vectors = steering_vectors.to(args.device)
    steering_vectors = steering_vectors * args.alpha
    
    if args.property == "therm":
        scoring_vec = steering_vectors.clone().unsqueeze(0)  
    elif args.property == "sol":
        scoring_vec = pos_steering_vectors.clone().to(args.device).unsqueeze(0)  
    new_seqs = [[] for _ in range(args.round)]

    with torch.no_grad():
        for seq in tqdm(islice(org_seqs, args.n), total=args.n):
            seq_token = torch.tensor(tokenizer.encode(seq), dtype=torch.int64, device=args.device)
            prev_seq_token = seq_token.clone()
            prev_mut_sites = set()

            for r in range(args.round):
                features = get_tokenwise_representations(prev_seq_token.unsqueeze(0), model)[0]
                if args.property == "therm":
                    related_score = F.cosine_similarity(features, scoring_vec, dim=-1).cpu().numpy()
                    mut_sites = topk_intersection_indices(related_score[:, 2], related_score[:, 3], args.T + len(prev_mut_sites))

                elif args.property == "sol":
                    features = features - features.mean(dim=0)
                    related_score = F.cosine_similarity(features, scoring_vec, dim=-1).cpu().numpy()
                    mut_sites = topk_intersection_indices(related_score[:, 28], related_score[:, 29], args.T + len(prev_mut_sites))

                mut_sites = [m for m in mut_sites if m not in prev_mut_sites][:args.T]
                prev_mut_sites.update(mut_sites)
                mut_sites_tensor = torch.LongTensor(mut_sites).to(args.device) + 1

                masked_seq = prev_seq_token.clone()
                masked_seq[mut_sites_tensor] = SEQUENCE_MASK_TOKEN
                new_seq_token = pred_tokens(masked_seq, model, steering_vectors, original_prediction=prev_seq_token, temperature=0.0)
                prev_seq_token = masked_seq.clone()
                prev_seq_token[mut_sites_tensor] = new_seq_token[mut_sites_tensor]

                new_seqs[r].append(decode_sequence(prev_seq_token, tokenizer))
                
    seqs_list = []
    epoch_list = []
    for r in range(args.round):
        seqs_list.extend(new_seqs[r])
        epoch_list.extend([r+1] * len(new_seqs[r]))

    res_df = pd.DataFrame({'sequence': seqs_list, 'epoch': epoch_list})
    if args.output_file is not None:
        res_df.to_csv(args.output_file, index=False)
        print('Generated sequences saved to:', args.output_file)

