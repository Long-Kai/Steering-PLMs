import argparse
import tqdm
import pandas as pd
import numpy as np
import torch

from module.steerable_prollama import SteerableLLaMA

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Path to a local prollama checkpoint directory.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num_gpus", type=str, default="1") 
    parser.add_argument("--max_gpu_memory", type=int, default=31)

    parser.add_argument('--n', type = int, default = 1000, help="number of sequences to generate")
    parser.add_argument('--temperature', type = float, default = 1.0)
    parser.add_argument('--top_p', type = float, default = 0.9)

    parser.add_argument('--property', type = str, default = "therm", help="therm, sol")
    parser.add_argument('--output_file', type = str, default=None, help="path to save the generated sequences")

    parser.add_argument('--steering', action='store_true', default=False, help="whether to use steering vectors. If False, generate sequences without steering.")
    parser.add_argument('--alpha', type = float, default = 1.0)
    parser.add_argument('--sv_from', type=str, default="saved_steering_vectors", help="folder path to load steering vectors from")
    parser.add_argument("--steer_only_first_token", action="store_true")

    args = parser.parse_args()

    # load steering vector
    if args.steering:
        steering_vectors_path = f"{args.sv_from}/Prollama_{args.property}_head_steering_vectors.pt"
        pos_steering_vectors, neg_steering_vectors = torch.load(steering_vectors_path)
        head_svs = pos_steering_vectors - neg_steering_vectors
        head_svs = args.alpha * head_svs.to(args.device)

        steering_vectors_path = f"{args.sv_from}/Prollama_{args.property}_mlp_steering_vectors.pt"
        pos_steering_vectors, neg_steering_vectors = torch.load(steering_vectors_path)
        mlp_svs = pos_steering_vectors - neg_steering_vectors
        mlp_svs = args.alpha * mlp_svs.to(args.device)

        steering_vectors = (head_svs, mlp_svs)
    else:
        steering_vectors = None

    plm = SteerableLLaMA(args.model_dir, args.device, args.max_gpu_memory, num_gpus=int(args.num_gpus), steering_vectors=steering_vectors, steer_only_first_token=args.steer_only_first_token)
    plm.set_stop_words([">"])

    def extract_seq(output):
        split = output.split('Seq=')[-1]
        tmp = ""
        for ch in split:
            if ch.isupper():
                tmp += ch
        return tmp
    
    seqs = []
    with torch.no_grad():
        for data in tqdm.tqdm(range(args.n)):
            raw_input_text = '[Generate by superfamily]Superfamily=<Lysozyme-like domain superfamily>Seq=<'

            generation_output = plm.generate(raw_input_text)
            seq = generation_output.split('Seq=<')[1].split('>')[0]
            seqs.append(seq)
            
    res_df = pd.DataFrame({'sequence': seqs})
    if args.output_file is not None:
        res_df.to_csv(args.output_file, index=False)
        print('Generated sequences saved to:', args.output_file)
