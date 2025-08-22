import torch
from tqdm import tqdm
import pandas as pd
import argparse
from baukit import TraceDict
import llama


def construct_triplet_dataset(data_path, theshold_pos, theshold_neg, num_data=None):
    instruction = '[Generate by superfamily]'
    inp = 'Superfamily=<Lysozyme-like domain superfamily>'

    pos_dataset = []
    neg_dataset = []

    df = pd.read_csv(data_path)
    pos_seqs = df['sequence'][df['score']>=theshold_pos].to_list()
    neg_seqs = df['sequence'][df['score']<=theshold_neg].to_list()

    if num_data is not None:
        pos_seqs = pos_seqs[:num_data]
        neg_seqs = neg_seqs[:num_data]

    for seq in pos_seqs:
        output = 'Seq=<' + seq
        pos_dataset.append({'instruction': instruction, 'input': inp, 'output': output, 'label': 1})

    for seq in neg_seqs:
        output = 'Seq=<' + seq
        neg_dataset.append({'instruction': instruction, 'input': inp, 'output': output, 'label': 0})

    return pos_dataset, neg_dataset


def get_llama_activations_bau(model, prompt, device): 
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+MLPS) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states

def tokenized_pro(dataset, tokenizer): 
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):

        instruction = dataset[i]['instruction']
        input = dataset[i]['input']
        output = dataset[i]['output']
        label = dataset[i]['label']

        prompt = f"{instruction}{input}{output}"
        prompt = tokenizer(prompt, return_tensors="pt").input_ids

        all_prompts.append(prompt)
        all_labels.append(label)

    return all_prompts, all_labels

def extract_activation(dataset, tokenizer, model, device):
    print("Tokenizing")
    prompts, _ = tokenized_pro(dataset, tokenizer)

    all_head_wise_activations_last = []
    all_mlp_wise_activations_last = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        _, head_wise_activations, mlp_wise_activations = get_llama_activations_bau(model, prompt, device)

        all_head_wise_activations_last.append(head_wise_activations[:,-1,:])
        all_mlp_wise_activations_last.append(mlp_wise_activations[:,-1,:])

    return all_head_wise_activations_last, all_mlp_wise_activations_last


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Path to a local prollama checkpoint directory.")
    parser.add_argument('--device', type=str, default="cuda")

    parser.add_argument('--property', type=str, choices=['therm', 'sol'], default='therm')
    
    parser.add_argument('--data_path', type=str, required=True, help="Path to the CSV file containing sequences and property score.")
    parser.add_argument('--theshold_pos', type=float, required=True, help="Threshold for positive data set.")
    parser.add_argument('--theshold_neg', type=float, required=True, help="Threshold for negative data set.")
    parser.add_argument('--num_data', type=int, default=None, help="Number of sequences to process. If None, processes all sequences.")
    parser.add_argument('--save_folder', type=str, default="saved_steering_vectors", help="Folder path to save steering vectors.")

    args = parser.parse_args()
    
    positive_dataset, negative_dataset = construct_triplet_dataset(args.data_path, args.theshold_pos, args.theshold_neg, args.num_data)

    model = llama.LlamaForCausalLM.from_pretrained(
        args.model_dir,
        low_cpu_mem_usage=True,
        device_map='auto',
        quantization_config=None
    )
    tokenizer = llama.LlamaTokenizer.from_pretrained(args.model_dir)
    model.eval()

    all_head_wise_activations_last, all_mlp_wise_activations_last = extract_activation(positive_dataset, tokenizer, model, args.device)
    pos_head_sv = torch.stack(all_head_wise_activations_last).mean(dim=0)
    pos_mlp_sv = torch.stack(all_mlp_wise_activations_last).mean(dim=0)

    all_head_wise_activations_last, all_mlp_wise_activations_last = extract_activation(negative_dataset, tokenizer, model, args.device)
    neg_head_sv = torch.stack(all_head_wise_activations_last).mean(dim=0)
    neg_mlp_sv = torch.stack(all_mlp_wise_activations_last).mean(dim=0)

    print("Saving activations")
    torch.save((pos_head_sv, neg_head_sv), f"{args.save_folder}/Prollama_{args.property}_head_steering_vectors.pt")

    torch.save((pos_mlp_sv, neg_mlp_sv), f"{args.save_folder}/Prollama_{args.property}_mlp_steering_vectors.pt")

  





