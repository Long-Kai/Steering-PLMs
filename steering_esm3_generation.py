import pandas as pd
import argparse
import numpy as np
import math
from tqdm import tqdm
import types
import torch
import torch.nn.functional as F

from module.steerable_esm3 import steering_forward, esm3_steering_forward
from utils.esm3_utils import load_esm3_model, generate_sequences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =========================== must be provided ===============================
    parser.add_argument('--ref_data_path', type = str, required=True, help="path to the reference data file ")

    #=========================== default is set ======================================
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")

    parser.add_argument('--n', type = int, default = 1000, help="number of sequences to generate")
    parser.add_argument('--temperature', type = float, default = 1.0)
    parser.add_argument('--top_p', type = float, default = 0.9)
    parser.add_argument('--mask_ratio', type = float, default = 0.1, help="ratio of masked tokens every round for generation")
    parser.add_argument('--property', type = str, default = "therm", help="therm, sol")
    parser.add_argument('--output_file', type = str, default=None, help="path to save the generated sequences")

    parser.add_argument('--steering', action='store_true', default=False, help="whether to use steering vectors. If False, generate sequences without steering.")
    parser.add_argument('--alpha', type = float, default = 1.0)
    parser.add_argument('--sv_from', type=str, default="saved_steering_vectors", help="folder path to load steering vectors from")
    #============================================================================
    args = parser.parse_args() 

    # load reference data
    df = pd.read_csv(args.ref_data_path)
    ref_seqs = df['sequence'].to_list()
    
    # load model
    model, tokenizer = load_esm3_model()
    model.transformer.steering_forward = types.MethodType(steering_forward,  model.transformer)
    model.steering_forward = types.MethodType(esm3_steering_forward, model)
    
    # load steering vector
    if args.steering:
        steering_vectors_path = f"{args.sv_from}/ESM3_{args.property}_steering_vectors.pt"
        pos_steering_vectors, neg_steering_vectors = torch.load(steering_vectors_path)
        steering_vectors = pos_steering_vectors - neg_steering_vectors
        steering_vectors = steering_vectors.to(args.device)
        steering_vectors = steering_vectors * args.alpha
    else:
        steering_vectors = None
    
    gen_seqs = []
    for i in tqdm(range(args.n)):
        seq = ref_seqs[i % len(ref_seqs)]
        seq_token = tokenizer.encode(seq)
        seq_token = torch.tensor(seq_token, dtype=torch.int64).to(args.device)
        new_seq = generate_sequences(seq_token, model, steering_vectors, args.mask_ratio, tokenizer, temperature=args.temperature, top_p=args.top_p)
        gen_seqs.append(new_seq)

    res_df = pd.DataFrame({'sequence': gen_seqs})
    if args.output_file is not None:
        res_df.to_csv(args.output_file, index=False)
        print('Generated sequences saved to:', args.output_file)

