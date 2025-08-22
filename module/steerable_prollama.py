import torch
import torch.nn.functional as F
from einops import rearrange

from transformers.utils import add_start_docstrings
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, STOPPING_CRITERIA_INPUTS_DOCSTRING
from transformers import GenerationConfig
import llama
import pdb

class SteeringLayer(torch.nn.Module):
    def __init__(self, steering_vector, steer_only_first_token=False):
        super(SteeringLayer, self).__init__()
        self.steer_only_first_token=steer_only_first_token
        self.steering_vec = steering_vector
        
    def forward(self, x):
        input_dtype = x.dtype
        inputs = x.clone()
        bs, L, d = inputs.size()

        if self.steer_only_first_token and L > 1:
            return inputs.type(input_dtype)

        inputs = rearrange(inputs, 'b l d -> d (b l)')
        # Get the norm for this layer's output
        norm = torch.norm(inputs.float(),dim=-1).unsqueeze(-1)

        inputs =  inputs + self.steering_vec.unsqueeze(-1)
        new_norm = torch.norm(inputs.float(),dim=-1).unsqueeze(-1) 
        inputs = inputs * norm
        inputs = inputs / new_norm.clamp_(1e-5)
        inputs = rearrange(inputs, 'd (b l) -> b l d', b=bs, l=L)
        
        return inputs.type(input_dtype)
    
class steerable_model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, steering_vectors, steer_only_first_token=False):
        head_svs, mlp_svs = steering_vectors

        for i in range(len(self.model.model.layers)):
                self.model.model.layers[i].self_attn.head_out = torch.nn.Sequential(self.model.model.layers[i].self_attn.head_out, SteeringLayer(head_svs[i], steer_only_first_token=steer_only_first_token)) 
                self.model.model.layers[i].mlp = torch.nn.Sequential(self.model.model.layers[i].mlp, SteeringLayer(mlp_svs[i],steer_only_first_token=steer_only_first_token)) 
            
        return self.model

    def remove_adapter(self): 
        for i in range(0, len(self.model.transformer.h)):
            self.model.transformer.h[i].self_attn.head_out = self.model.transformer.h[i].self_attn.head_out[0]
            self.model.transformer.h[i].mlp = self.model.transformer.h[i].mlp[0]

class LLamaStoppingCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the model generates specific token in stop_token_id.
    """
    def __init__(self, list_token_ids_sequence):
        self.token_ids_sequences = []
        self.lengths = []
        for token_ids_sequence in list_token_ids_sequence:
            self.token_ids_sequences.append(torch.tensor(token_ids_sequence, dtype=torch.long))
            self.lengths.append(len(token_ids_sequence))
        
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # check the final {self.length} tokens
        stop = False
        for token_ids_sequence, length in zip(self.token_ids_sequences, self.lengths):
            if input_ids.shape[-1] < length:
                continue
            else:
                if bool(torch.all(input_ids[0, -length:] == token_ids_sequence.to(input_ids.device))):
                    stop = True
                    break
        return stop
    
class SteerableLLaMA:
    def __init__(self, model_dir, device="cuda", max_gpu_memory=31, num_gpus=-1,  steering_vectors=None, steer_only_first_token=False):
        """
        Args:
            model_dir (str): folder storing the pretrained llama model.
            device (str): used device. Defaults to `cuda`.
            max_gpu_memory (int, optional): max gpu memory. Defaults to 31.
            num_gpus (int, optional): number of used gpus for base model. Defaults to -1 (auto).  
            steering_vectors (tuple, optional): steering vectors for steering the model. Defaults to None.
            steer_only_first_token (bool, optional): whether to steer only the first token. Defaults to False.
        """
        self.device = device
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_dir, num_gpus)
        
        if steering_vectors is not None:
            while True:
                try:
                    steerable_model(self.model).remove_adapter()
                except:
                    break

            updated_wrapper = steerable_model(self.model)
            _ = updated_wrapper.get_model(steering_vectors, steer_only_first_token=steer_only_first_token)
            print('Steering Layer have been added!\n') 


    def load_model(self, model_dir, num_gpus, start_id=0):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{model_dir}/offload"} # must be bfloat16 for ProLLaMA
            
            if num_gpus == -1:
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if torch.cuda.device_count() != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(start_id, start_id + num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = llama.LlamaTokenizer.from_pretrained(model_dir)
        model = llama.LlamaForCausalLM.from_pretrained(model_dir, low_cpu_mem_usage=True, **kwargs)
        if self.device == "cuda" and num_gpus == 1:
            model.cuda()
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids)
        self.stopping_criteria.append(LLamaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text=None, input_ids=None, max_new_tokens=512, top_p=0.9, top_k=40, temperature=1.0, verbose=False):
        with torch.no_grad():
            input_ids = self.tokenizer(input_text, return_tensors="pt")
            generation_configs = GenerationConfig(
                                    temperature=temperature,
                                    top_k=top_k,
                                    top_p=top_p,
                                    do_sample=True,
                                    num_beams=1,
                                    repetition_penalty=1.2,
                                    max_new_tokens=max_new_tokens,
                                )      
            outputs = self.model.generate(
                                input_ids = input_ids["input_ids"].to(self.device),
                                attention_mask = input_ids['attention_mask'].to(self.device),
                                eos_token_id=self.tokenizer.eos_token_id,
                                pad_token_id=self.tokenizer.pad_token_id,
                                generation_config = generation_configs,
                                output_attentions=False
                            )

            # skip the tokens in the input prompt
            ######################################################
            gen_sequences = outputs[0]
            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)
            ######################################################
            
            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

        if self.device:
            torch.cuda.empty_cache()

        return output_str