import torch
import torch.nn.functional as F
import math
import esm

from utils.gen_utils import sample_top_p

def get_esm2_model_name(model_size):
    """
    Returns the ESM-2 model name based on the specified size (model name abbr.).
    Args: model_size (str): Size of the ESM-2 model ('150M', '650M', or '3B').
    Returns: str: Corresponding ESM-2 model name.
    """
    model_name_dict = {
        "150M": "esm2_t30_150M_UR50D",
        "650M": "esm2_t33_650M_UR50D",
        "3B": "esm2_t36_3B_UR50D"
    }
    
    if model_size in model_name_dict:
        return model_name_dict[model_size]
    else:
        raise ValueError(f"Unknown model size: {model_size}")

def get_esm2_layer_and_feature_dim(model_name):
    """
    Returns the number of layers and feature dimension for the specified ESM-2 model.
    Args: model_name (str): Name of the ESM-2 model (e.g., 'esm2_t6_8M_UR50D').
    Returns: tuple: (num_layers, feature_dim)
    """
    model_info = {
        "150M": (30, 640),
        "650M": (33, 1280),
        "3B": (36, 2560)
    }
    
    if model_name in model_info:
        return model_info[model_name]
    else:
        raise ValueError(f"Unknown ESM-2 model name: {model_name}")

def load_esm2_model(model_name, device='cuda', ckpt_path=None):
    """
    Loads the specified ESM-2 model and its alphabet.
    Args:
        model_name (str): Name of the ESM-2 model to load (e.g., 'esm2_t6_8M_UR50D').
        device (str): Device to load the model on ('cuda' or 'cpu').
        ckpt_path (str, optional): Path to a local checkpoint file. If provided, loads the model from this path.
    Returns:
        model: Loaded ESM-2 model.
        alphabet: ESM-2 alphabet object.
    """

    if ckpt_path is not None:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(ckpt_path)
    else:
        model, alphabet = esm.pretrained.load_model_and_alphabet(get_esm2_model_name(model_name))
    model = model.to(device)
    model.eval()
    model.token_dropout = False
    return model, alphabet

def extract_esm2_features(seqs, model, alphabet, n_layer, batch_size=1, device='cuda'):
    """
    Extracts ESM-2 features for a list of sequences from specified model layers.
    Args:
        seqs (list of str): Protein sequences.
        model: ESM-2 model.
        alphabet: ESM-2 alphabet object.
        layers (list of int): Layer indices to extract representations from.
        batch_size (int): Number of sequences per batch.
    Returns:
        torch.Tensor: Representations of shape (num_layers, num_seqs, feature_dim)
    """
    batch_converter = alphabet.get_batch_converter()
    # Prepare a list for each layer to collect representations
    layers = list(range(n_layer))
    layer_reps = [[] for _ in layers]

    # Process sequences in batches
    for start in range(0, len(seqs), batch_size):
        seq_batch = seqs[start:start + batch_size]
        x_batch = [("protein", seq) for seq in seq_batch]
        _, _, batch_tokens = batch_converter(x_batch)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens.to(device), repr_layers=layers)

        # Collect mean representations for each sequence and layer
        for layer_idx, layer in enumerate(layers):
            token_reps = results["representations"][layer]
            for seq_idx, seq_len in enumerate(batch_lens):
                # Exclude BOS/EOS tokens
                rep = token_reps[seq_idx, 1:seq_len-1].mean(0).cpu()
                layer_reps[layer_idx].append(rep)

    # Stack representations: (num_layers, num_seqs, feature_dim)
    stacked = torch.stack([torch.stack(reps) for reps in layer_reps])
    return stacked

def get_average_representation(tokens, model, n_layer):
    """
    Computes the average representation of a sequence from last layer of the ESM-2 model.
    Args:
        tokens (torch.Tensor): Input sequence tokens of shape (batch_size, seq_length).
        model: ESM-2 model.
        n_layer (int): Layer index to extract the representation from.
    Returns:
        torch.Tensor: Average representation of shape (batch_size, feature_dim).
    """

    with torch.no_grad():
        outputs = model(tokens=tokens, repr_layers=[n_layer])
    return outputs['representations'][n_layer][:, 1:-1].mean(dim=1)

def decode(vocab, seq_enc, onehot=True):
    if onehot:
        seq_enc = seq_enc.argmax(-1)
    assert seq_enc.dim() == 2
    seq_enc = seq_enc.cpu()
    seqs = [
        ''.join([vocab.get_tok(c) for c in _seq]) for _seq in seq_enc
    ]
    return seqs

def pred_tokens(tokens, model, steering_vectors=None, original_prediction=None, temperature=0.0, top_p=0.9):
    with torch.no_grad():
        if steering_vectors is not None:
            outputs = model.steering_forward(tokens=tokens, steering_vectors=steering_vectors)
        else:
            outputs = model(tokens=tokens)

    logits = outputs['logits']

    if original_prediction is not None:
        mask  = F.one_hot(original_prediction[0], logits.size(-1))[:, 4:24].to(logits. device)
        
    logits = logits[0, :, 4:24]

    if original_prediction is not None:
        logits = logits + mask.float() * -1e8 # add negative logits to avoid predicting original tokens

    if temperature > 0.0:
        probs = torch.softmax(logits / temperature, dim=-1)
        pred_seq = sample_top_p(probs, top_p)
    else:
        pred_seq = torch.argmax(logits, dim=-1)    
    pred_seq = pred_seq + 4

    pred_seq[0] = tokens[0, 0]
    pred_seq[-1] = tokens[0,-1]

    return pred_seq

def generate_sequences(tokens, model, steering_vectors, masked_ratio, alphabet, temperature=0.0, top_p=0.9):
    mask_idx = alphabet.mask_idx
    tokens = tokens.clone()
    length = tokens.size(1) - 2
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
        seq_token[0, mask_positions] = mask_idx
        new_seq = pred_tokens(seq_token, model, steering_vectors, temperature=temperature, top_p=top_p)
        tokens[0, mask_positions] = new_seq[mask_positions]

    return decode(alphabet, tokens[:, 1:-1], onehot=False)[0]