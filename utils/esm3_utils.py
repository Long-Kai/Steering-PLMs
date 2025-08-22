import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from esm3.pretrained import ESM3_sm_open_v0
from esm3.tokenization import get_model_tokenizers
from esm3.utils.constants.esm3 import SEQUENCE_PAD_TOKEN, SEQUENCE_MASK_TOKEN
from esm3.utils.decoding import decode_sequence

from utils.gen_utils import sample_top_p

def get_esm3_layer_and_feature_dim():
    return (48, 1536)

def load_esm3_model(device='cuda'):
    model = ESM3_sm_open_v0(device)
    model = model.to(device)
    model.eval()
    tokenizer = get_model_tokenizers().sequence
    return model, tokenizer

def extract_esm3_features(seqs, model, tokenizer, n_layer, batch_size=1, device='cuda'):
    layer_reps = [[] for _ in range(n_layer)]

    for start in range(0, len(seqs), batch_size):
        seq_batch = seqs[start:start + batch_size]
        x_batch = [tokenizer.encode(seq) for seq in seq_batch]
        batch_lens = [len(x) for x in x_batch]

        # Pad sequences
        sequence_tokens = pad_sequence(
            [torch.tensor(x, dtype=torch.int64) for x in x_batch],
            batch_first=True,
            padding_value=SEQUENCE_PAD_TOKEN
        )

        with torch.no_grad():
            _, representations = model(sequence_tokens=sequence_tokens.to(device), return_representations=True)

        for layer in range(n_layer):
            token_reps = representations[layer]
            for tokens, tokens_len in zip(token_reps, batch_lens):
                # Exclude special tokens at start and end
                layer_reps[layer].append(tokens[1:tokens_len-1].mean(0).cpu())

    # Stack representations: (num_layers, num_seqs, feature_dim)
    return torch.stack([torch.stack(reps) for reps in layer_reps])

def get_tokenwise_representations(tokens, model):
    with torch.no_grad():
        _, representations = model(sequence_tokens=tokens, return_representations=True)
    
    return torch.stack(representations).permute(1, 2, 0, 3)[:, 1:-1] # batch size x sequence length x num layers x feature dim

def get_average_representation(tokens, model):
    with torch.no_grad():
        _, representations = model(sequence_tokens=tokens, return_representations=True)
    return representations[-1][:, 1:-1].mean(dim=1)


def pred_tokens(tokens, model, steering_vectors=None, original_prediction=None, temperature=0.0, top_p=0.9):
    with torch.no_grad():
        if steering_vectors is not None:
            outputs = model.steering_forward(sequence_tokens=tokens.unsqueeze(0), steering_vectors=steering_vectors)
        else:
            outputs = model(sequence_tokens=tokens.unsqueeze(0))

    logits = outputs.sequence_logits

    if original_prediction is not None:
        mask  = F.one_hot(original_prediction, logits.size(-1))[:, 4:24].to(logits. device)
        
    logits = logits[0, :, 4:24]

    if original_prediction is not None:
        logits = logits + mask.float() * -1e8 # add negative logits to avoid predicting original tokens

    if temperature > 0.0:
        probs = torch.softmax(logits / temperature, dim=-1)
        pred_seq = sample_top_p(probs, top_p)
    else:
        pred_seq = torch.argmax(logits, dim=-1)    
    pred_seq = pred_seq + 4

    pred_seq[0] = tokens[0]
    pred_seq[-1] = tokens[-1]

    return pred_seq

def generate_sequences(tokens, model, steering_vectors, masked_ratio, tokenizer, temperature=0.0, top_p=0.9):
    mask_idx = SEQUENCE_MASK_TOKEN
    tokens = tokens.clone()
    length = tokens.size(0) - 2
    candidate_sites = list(range(length))
    rounds = math.ceil(1.0 / masked_ratio)

    for _ in range(rounds):
        mask_size = min(math.ceil(length * masked_ratio), len(candidate_sites))
        if mask_size == 0:
            break

        indices = torch.randperm(len(candidate_sites))[:mask_size]
        mask_positions = torch.tensor([candidate_sites[i] for i in indices]) + 1  # +1 for offset
        candidate_sites = [site for i, site in enumerate(candidate_sites) if i not in indices]

        seq_token = tokens.clone()
        seq_token[mask_positions] = mask_idx
        new_seq = pred_tokens(seq_token, model, steering_vectors, temperature=temperature, top_p=top_p)
        tokens[mask_positions] = new_seq[mask_positions]

    return decode_sequence(tokens, tokenizer)