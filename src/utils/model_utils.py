import torch
import numpy as np
import nnsight
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import os
import random
from typing import *


def load_gpt_model_and_tokenizer(model_name:str, device='cuda'):
    """
    Loads a huggingface model and its tokenizer

    Parameters:
    model_name: huggingface name of the model to load (e.g. GPTJ: "EleutherAI/gpt-j-6B", or "EleutherAI/gpt-j-6b")
    device: 'cuda' or 'cpu'
    
    Returns:
    model: huggingface model
    tokenizer: huggingface tokenizer
    MODEL_CONFIG: config variables w/ standardized names
    
    """
    assert model_name is not None

    print("Loading: ", model_name)

    if model_name == 'gpt2-xl':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
    elif 'gpt-j' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(device)

    elif 'gpt-neox' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        
    elif 'llama' in model_name.lower() or "alpaca" in model_name.lower():
        if '70b' in model_name.lower():
            # use quantization. requires `bitsandbytes` library
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config
            )
        else:
            if '7b' in model_name.lower():
                model_dtype = torch.float16
                
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)

    elif 'mistral' in model_name.lower():
        print(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    else:
        raise NotImplementedError("Still working to get this model available!")
    
    MODEL_CONFIG = get_model_config(model_name, model)

    return model, tokenizer, MODEL_CONFIG

def get_model_config(model_name:str, model):
    """
    Parameters: 
    model_name: huggingface name of the model to load (e.g. GPTJ: "EleutherAI/gpt-j-6B", or "EleutherAI/gpt-j-6b")
    model: instantiated model (either HF or nnsight)

    Returns: 
    MODEL_CONFIG: config variables w/ standardized names
    """

    if model_name == 'gpt2-xl':
        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)]}
        
    elif 'gpt-j' in model_name.lower():
        MODEL_CONFIG={"n_heads":model.config.n_head,
                      "n_layers":model.config.n_layer,
                      "resid_dim":model.config.n_embd,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'transformer.h.{layer}.attn.out_proj' for layer in range(model.config.n_layer)],
                      "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.config.n_layer)]}

    elif 'gpt-neox' in model_name.lower():
        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim": model.config.hidden_size,
                      "name_or_path":model.config.name_or_path,
                      "attn_hook_names":[f'gpt_neox.layers.{layer}.attention.dense' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'gpt_neox.layers.{layer}' for layer in range(model.config.num_hidden_layers)]}
        
    elif 'llama' in model_name.lower() or "alpaca" in model_name.lower():
        MODEL_CONFIG={"n_heads":model.config.num_attention_heads,
                      "n_layers":model.config.num_hidden_layers,
                      "resid_dim":model.config.hidden_size,
                      "name_or_path":model.config._name_or_path,
                      "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
                      "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)]}
    elif 'mistral' in model_name.lower():
        MODEL_CONFIG={
            "n_heads":model.config.num_attention_heads,
            "n_layers":model.config.num_hidden_layers,
            "resid_dim":model.config.hidden_size,
            "name_or_path":model.config._name_or_path,
            "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
            "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)]
        }
    else:
        raise NotImplementedError("Still working to get this model available!")
    
    return MODEL_CONFIG



def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_nnsight_model(model_name:str, device='cuda', load_locally=True):
    """
    Loads an nnsight LanguageModel version

    Parameters:
    model_name: HuggingFace model name/ID
    device: cuda or cpu
    load_locally: whether to instantiate the model upon loading

    Returns:
    model: nnsight model
    tokenizer: nnsight tokenizer
    MODEL_CONFIG: config variables w/ standardized names
    """
    
    model = nnsight.LanguageModel(model_name, device_map=device, dispatch=load_locally).to(torch.bfloat16)
    model_config = get_model_config(model_name, model)

    return model, model.tokenizer, model_config

    

def set_requires_grad(module:torch.nn.Module, requires_grad:bool):
    """
    Sets the requires_grad attribute for all parameters of a torch module. Does so in place

    Parameters: 
    module: torch module with parameters
    requires_grad: whether to set grad to True or False
    """
    
    assert isinstance(requires_grad, bool)

    for parameter in module.parameters():
        parameter.requires_grad = requires_grad

def count_parameters(module):
    """Count parameters of a module that require gradients"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
        