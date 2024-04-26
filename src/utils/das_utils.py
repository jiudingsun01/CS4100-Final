from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader

import os
import torch
from tqdm import tqdm
import numpy as np
from pyvene.models.configuration_intervenable_model import RepresentationConfig, IntervenableConfig
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.interventions import BoundlessRotatedSpaceIntervention
from pyvene.models.basic_utils import set_seed, count_parameters, sigmoid_boundary

from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *


def get_rotation_mask(intervenable_model: IntervenableModel):
    
    assert len(intervenable_model.interventions.keys()) == 1
    
    intervention_key = list(intervenable_model.interventions.keys())[0]
    
    intervention = intervenable_model.interventions[intervention_key][0]
    intervention_boundaries = torch.clamp(intervention.intervention_boundaries, 1e-3, 1)
    boundary_mask = sigmoid_boundary(
        intervention.intervention_population.repeat(1, 1),
        0.0,
        intervention_boundaries[0] * int(intervention.embed_dim),
        intervention.temperature
    )
    
    return boundary_mask

def compute_rotation_mask_sparsity(intervenable_model: IntervenableModel):
        
        rotation_mask = get_rotation_mask(intervenable_model)
        return (rotation_mask.sum() / rotation_mask.numel()).item()
    

def compute_metrics(eval_preds, eval_labels, generate_output=False):
    
    total_count = 0
    correct_count = 0
    
    if generate_output:
        outputs = []
        gts = []
        
        
        
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        
        for i in range(eval_label.shape[0]):
            label_idxs = eval_label[i].ne(IGNORE_INDEX).nonzero().squeeze(-1)
                        
            actual_test_labels = eval_label[i][label_idxs].tolist()
            pred_test_labels = [eval_pred[i][idx].argmax(dim=-1).item() for idx in label_idxs]
            
            correct = actual_test_labels==pred_test_labels # uncomment it to evaluate all tokens
            
            if generate_output:
                outputs.append(pred_test_labels)
                gts.append(actual_test_labels)
                        
            total_count += 1
            if correct:
                correct_count += 1
                
    return_dict = {"accuracy": round(correct_count/total_count, 2)} 
    if generate_output:
        return_dict["outputs"] = outputs
        return_dict["labels"] = gts
        
    return return_dict


def evaluate(intervenable_model, dataloader, device="cuda", intervene=False, corrupt=False, generate_output=False):
    
    with torch.no_grad():
        
        eval_labels = []
        eval_preds = []
        
        for step, inputs in enumerate(tqdm(dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                    
            if not intervene:
                
                if corrupt:
                    outputs = intervenable_model.model(
                        input_ids=inputs['base_input_ids'],
                        labels=inputs['base_labels'],
                        attention_mask=inputs['base_attention_mask']
                    )
                    eval_labels += [inputs['base_labels']]
                else:
                    outputs = intervenable_model.model(
                        input_ids=inputs['source_input_ids'],
                        labels=inputs['source_labels'],
                        attention_mask=inputs['source_attention_mask']
                    )
                    eval_labels += [inputs['source_labels'].detach().cpu()]
                
            else:           
                source2base = ([[[idx] for idx in inputs["source_predictive_token_idxs"].tolist()]], [[[idx] for idx in inputs["predictive_token_idxs"].tolist()]])
                
                _, outputs = intervenable_model(
                    {"input_ids": inputs["base_input_ids"], "attention_mask": inputs["base_attention_mask"]},
                    [{"input_ids": inputs["source_input_ids"], "attention_mask": inputs["source_attention_mask"]}],
                    {"sources->base": source2base}
                )
                eval_labels += [inputs['base_labels'].detach().cpu()]
                
            eval_preds += [outputs.logits.detach().cpu()]
        
        eval_metrics = compute_metrics(eval_preds, eval_labels, generate_output=generate_output)
        return eval_metrics


def simple_boundless_das_position_config(model_type, intervention_type, layer):
    intervenable_config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,             # layer
                intervention_type, # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            ),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    return intervenable_config


def process_dataset(dataset, model_config, tokenizer, n_shots, data_split, prefixes, separators, n_trials=None, ablation_method="zero_shot", draw_source_from_split=True):
    
    
    assert ablation_method in ["zero_shot", "noninformative", "none"]
    torch_dataset = []
    
    is_llama = 'llama' in model_config['name_or_path'] or 'alpaca' in model_config['name_or_path'] or 'mistral' in model_config['name_or_path']
    prepend_bos = not is_llama
    
    if n_trials is None:
        sample_idxs = range(len(dataset[data_split]))
    else:
        sample_idxs = np.random.choice(len(dataset[data_split]), n_trials, replace=True).tolist()
    
    for i in sample_idxs:
        
        data_pair = {}
        
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]  
        
        if not draw_source_from_split:
            word_pairs_test = dataset[data_split][i]
            source_test_word_pair = dataset[data_split][np.random.choice(len(dataset[data_split]), 1, replace=False)]
        else:
            source_test_word_pair = dataset[data_split][i]
            word_pairs_test = dataset[data_split][np.random.choice(len(dataset[data_split]), 1, replace=False)]
            
        if type(prefixes) == dict and type(separators) == dict:
            prefix = prefixes
            separator = separators
        elif type(prefixes) == list and type(separators) == list:
            rand_idx = np.random.choice(len(prefixes))
            prefix = prefixes[rand_idx]
            separator = separators[rand_idx]
        else:
            raise ValueError("prefixes and separators should be either both list or dict")
        
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=source_test_word_pair, prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=prefix, separators=separator)
        
        query = prompt_data['query_target']['input']
        target = prompt_data['query_target']['output']
        
        source_token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
        
        source_batch = preprocess([prompt_string], [target], tokenizer)
        
        data_pair["source_input_ids"] = source_batch["input_ids"]
        data_pair["source_labels"] = source_batch["labels"]
        
        assert source_token_labels[-1][2] == "query_predictive_token"
        source_predictive_token_idxs = source_token_labels[-1][0]
        data_pair["source_predictive_token_idxs"] = source_predictive_token_idxs
        
        if ablation_method == "none":
            pass
        else:
            if ablation_method == "zero_shot":
                base_word_pairs = {'input':[], 'output':[]}
            elif ablation_method == "noninformative":
                base_word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]
            else:
                raise ValueError(f"ablation_method {ablation_method} is not supported.")
            
            ablation_prefix = {"input": prefix["input"], "output": prefix["output"], "instructions": ""}
            ablation_separator = {"input": separator["input"], "output": separator["output"], "instructions": ""}
            
            base_prompt_data = word_pairs_to_prompt_data(base_word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=True, prefixes=ablation_prefix, separators=ablation_separator)
            
            base_query = base_prompt_data['query_target']['input']
            base_target = base_prompt_data['query_target']['output']
            
            token_labels, base_prompt_string = get_token_meta_labels(base_prompt_data, tokenizer, base_query)
            
            base_batch = preprocess([base_prompt_string], [base_target], tokenizer)
            data_pair["base_input_ids"] = base_batch["input_ids"]
            data_pair["base_labels"] = base_batch["labels"]
            
            assert token_labels[-1][2] == "query_predictive_token"
            predictive_token_idxs = token_labels[-1][0]
            data_pair["predictive_token_idxs"] = predictive_token_idxs
            
        torch_dataset.append(data_pair)
            
    torch_dataset = Dataset.from_list(torch_dataset)
    torch_dataset.set_format(type='torch')
    return torch_dataset


def process_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, data_split, prefixes, separators, collate_fn, n_trials=None, ablation_method="zero_shot", shuffle=False, draw_source_from_split=True):
    
    torch_dataset = process_dataset(dataset, model_config, tokenizer, n_shots, data_split, prefixes, separators, n_trials, ablation_method, draw_source_from_split)
    torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return torch_dataloader


def process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, data_split, prefixes, separators, collate_fn, n_trials=None, ablation_method="zero_shot", draw_source_from_split=True):
    
    all_dataset = []
    
    use_task_specific_prefixes_separators = False
    if type(prefixes) == list and type(separators) == list:
        if type(prefixes[0]) == list and type(separators[0]) == list and len(prefixes) == len(separators) == len(datasets):
            use_task_specific_prefixes_separators = True
            print("Using task specific prefixes and separators...")
            
            
    for dataset in datasets:
        
        if use_task_specific_prefixes_separators:
            prefixes_ = prefixes[datasets.index(dataset)]
            separators_ = separators[datasets.index(dataset)]
        else:
            prefixes_ = prefixes
            separators_ = separators
        
        torch_dataset = process_dataset(dataset, model_config, tokenizer, n_shots, data_split, prefixes_, separators_, n_trials, ablation_method, draw_source_from_split)
        all_dataset.append(torch_dataset)
    
    all_dataset = concatenate_datasets(all_dataset)
    torch_dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return torch_dataloader


def load_intervention_weight(intervenable, intervenable_from):
        
    assert len(intervenable_from.interventions.keys()) == len(intervenable.interventions.keys()) == 1

    source_key = list(intervenable_from.interventions.keys())[0]
    target_key = list(intervenable.interventions.keys())[0]

    source_intervention = intervenable_from.interventions[source_key][0]

    temp_weight = source_intervention.temperature.data
    intervention_population_weight = source_intervention.intervention_population.data
    intervention_boundaries_weight = source_intervention.intervention_boundaries.data

    matrix_weight = source_intervention.rotate_layer.weight

    intervenable.interventions[target_key][0].temperature.data = temp_weight.clone().detach()
    intervenable.interventions[target_key][0].intervention_population.data = intervention_population_weight.clone().detach()
    intervenable.interventions[target_key][0].intervention_boundaries.data = intervention_boundaries_weight.clone().detach()
    intervenable.interventions[target_key][0].rotate_layer.weight = matrix_weight.clone().detach()
    
    return intervenable


def process_alpaca_eval_dataloader(dataset, tokenizer, collate_fn, batch_size, instruction, data_split="test", corrupt_base=True):
    return_dict = {}
    
    base_inputs, base_labels = [], []
    for base_dict in dataset[data_split]:
        if corrupt_base:
            base_input, base_label = load_alpaca_input_output(**base_dict, instruction="")
        else:
            base_input, base_label = load_alpaca_input_output(**base_dict, instruction=instruction)
        base_inputs.append(base_input)
        base_labels.append(base_label)
    
    base = preprocess(base_inputs, base_labels, tokenizer)
    
    # Find the indices of the last tokens for each input
    predictive_token_idxs = [len(label) - label.ne(IGNORE_INDEX).sum() for label in base["labels"]]
    
    return_dict["base_input_ids"] = base["input_ids"]
    return_dict["base_labels"] = base["labels"]
    return_dict["predictive_token_idxs"] = predictive_token_idxs
    
    source_inputs, source_labels = [], []
    for _ in range(len(dataset[data_split])):
        input_dict = random.choice(dataset["train"])
        source_input, source_label = load_alpaca_input_output(**input_dict, instruction=instruction)
        source_inputs.append(source_input)
        source_labels.append(source_label)
        
    source = preprocess(source_inputs, source_labels, tokenizer)
    
    # Find the indices of the last tokens for each input
    source_predictive_token_idxs = [len(label) - label.ne(IGNORE_INDEX).sum() for label in source["labels"]]
    
    return_dict["source_input_ids"] = source["input_ids"]
    return_dict["source_labels"] = source["labels"]
    return_dict["source_predictive_token_idxs"] = source_predictive_token_idxs
    
    dataset = Dataset.from_dict(return_dict)
    dataset.set_format(type='torch')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader
    
    
        

def process_alpaca_dataloaders(data_path, tokenizer, batch_size, test_split, collate_fn, n_trials=None):
    alpaca_dict = json.load(open(data_path, "r"))
    alpaca_dataset = Dataset.from_list(alpaca_dict)
    alpaca_dataset = alpaca_dataset.filter(lambda x: len(x["input"]) > 0 and len(x["output"]) > 0)    
    return_dict = {}
    
    source_inputs, source_labels = [], []
    for input_dict in alpaca_dataset:
        source_input, source_label = load_alpaca_input_output(**input_dict)
        source_inputs.append(source_input)
        source_labels.append(source_label)
    
    source = preprocess(source_inputs, source_labels, tokenizer)
    
    # Find the indices of the last tokens for each input
    source_predictive_token_idxs = [len(label) - label.ne(IGNORE_INDEX).sum() for label in source["labels"]]
    
    return_dict["source_input_ids"] = source["input_ids"]
    return_dict["source_labels"] = source["labels"]
    return_dict["source_predictive_token_idxs"] = source_predictive_token_idxs
    
    base_inputs, base_labels = [], []
    for input_dict in alpaca_dataset:
        input_dict["instruction"] = ""
        base_input, base_label = load_alpaca_input_output(**input_dict)
        base_inputs.append(base_input)
        base_labels.append(base_label)
    
    base = preprocess(base_inputs, base_labels, tokenizer)
    
    # Find the indices of the last tokens for each input
    predictive_token_idxs = [len(label) - label.ne(IGNORE_INDEX).sum() for label in base["labels"]]
    
    return_dict["base_input_ids"] = base["input_ids"]
    return_dict["base_labels"] = base["labels"]
    return_dict["predictive_token_idxs"] = predictive_token_idxs
    
    base_exceed_length, source_exceed_length = [], []
    for i in range(len(return_dict["source_input_ids"])):
        if return_dict["source_input_ids"][i].shape[0] > tokenizer.model_max_length:
            source_exceed_length.append(i)
    
    for i in range(len(return_dict["base_input_ids"])):
        if return_dict["base_input_ids"][i].shape[0] > tokenizer.model_max_length:
            base_exceed_length.append(i)
            
    redundent_idxs = set(source_exceed_length).union(set(base_exceed_length))
    
    print(f"Remove {len(redundent_idxs)} samples from the dataset due to exceeding the maximum length of the model ({tokenizer.model_max_length})")
    for key in return_dict.keys():
        return_dict[key] = [return_dict[key][i] for i in range(len(return_dict[key])) if i not in redundent_idxs]
    split_idx = int(len(return_dict["source_input_ids"]) * test_split)
    
    if test_split == 0:
        train_dataset = Dataset.from_dict(return_dict)
        
        if n_trials is not None:
            train_dataset = train_dataset.select(range(n_trials))
            
        train_dataset.set_format(type='torch')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        return train_dataloader, None
    
    train_dict = {k: v[split_idx:] for k, v in return_dict.items()}
    test_dict = {k: v[:split_idx] for k, v in return_dict.items()}
    
    train_dataset = Dataset.from_dict(train_dict)
    if n_trials is not None:
        train_dataset = train_dataset.select(range(n_trials))
    
    train_dataset.set_format(type='torch')
    
    test_dataset = Dataset.from_dict(test_dict)
    test_dataset.set_format(type='torch')
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, test_dataloader
    
    

    
    
    