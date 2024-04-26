import torch
from tqdm import tqdm
import numpy as np
import nnsight
import transformers
import copy
from typing import *
from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets
from utils.prompt_utils import word_pairs_to_prompt_data, get_token_meta_labels

IGNORE_INDEX = -100


# from pyvene https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/layers.py#L16 
class RotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = torch.empty(n, n)
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)

def sigmoid_boundary(_input, boundary_x, boundary_y, temperature):
    """Generate sigmoid mask"""
    return torch.sigmoid((_input - boundary_x) / temperature) * torch.sigmoid(
        (boundary_y - _input) / temperature
    )

def get_rotation_mask(subspace_proj):
    """
    """
    intervention_boundaries = torch.clamp(subspace_proj.intervention_boundaries, 1e-3, 1)
    boundary_mask = sigmoid_boundary(
        subspace_proj.intervention_population.repeat(1, 1),
        0.0,
        intervention_boundaries[0] * int(subspace_proj.embed_dim),
        subspace_proj.temperature
    )
    return boundary_mask


def get_rotation_mask_fixed_dimension(subspace_proj):
    return torch.sigmoid(subspace_proj.get_mask_boundaries() / subspace_proj.get_temperature())

def compute_rotation_mask_sparsity(subspace_proj):
    """
    """
    rotation_mask = get_rotation_mask(subspace_proj)
    return (rotation_mask.sum() / rotation_mask.numel()).item()

def compute_rotation_mask_sparsity_by_attentions(subspace_projs):
    """
    """
    
    total, used = 0, 0
    for layer in subspace_projs.keys():
        for idx in subspace_projs[layer].keys():
            rotation_mask = get_rotation_mask(subspace_projs[layer][idx])
            total += rotation_mask.numel()
            used += rotation_mask.sum()

    return used / total

# from pyvene https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/interventions.py#L298
class BoundlessRotatedSpaceIntervention(torch.nn.Module):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([1.0]), requires_grad=True
        )
        self.temperature = torch.nn.Parameter(torch.tensor(10.0))
        self.intervention_population = torch.nn.Parameter(
            torch.arange(0, self.embed_dim), requires_grad=False
        )

    def get_boundary_parameters(self):
        return self.intervention_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def set_intervention_boundaries(self, intervention_boundaries):
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([intervention_boundaries]), requires_grad=True
        )
        
    def forward(self, base, source, batch_size):
        # batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        intervention_boundaries = torch.clamp(self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(batch_size, 1),
            0.0,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature,
        )
        boundary_mask = (
            torch.ones_like(base)[:,0].unsqueeze(dim=-1).to(base.device) * boundary_mask
            # torch.ones(batch_size, device=base.device).unsqueeze(dim=-1) * boundary_mask
        )
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (
            1.0 - boundary_mask
        ) * rotated_base + boundary_mask * rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"BoundlessRotatedSpaceIntervention()"
    
    

class BoundlessRotatedSpaceFixedDimensionIntervention(torch.nn.Module):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.mask_boundaries = torch.nn.Parameter(
            torch.tensor([500.0 for _ in range(embed_dim)]), requires_grad=True
        )
        self.temperature = torch.nn.Parameter(torch.tensor(50.0))

    def get_mask_boundaries(self):
        return self.mask_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def set_mask_boundaries(self, mask_boundaries):
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor(mask_boundaries), requires_grad=True
        )
        
    def forward(self, base, source, batch_size):
        # batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        # mask_boundaries = torch.clamp(self.intervention_boundaries, 1e-3, 1)
        boundary_mask = torch.sigmoid(self.mask_boundaries / self.temperature)
        
        boundary_mask = (
            torch.ones_like(base)[:,0].unsqueeze(dim=-1).to(base.device) * boundary_mask
            # torch.ones(batch_size, device=base.device).unsqueeze(dim=-1) * boundary_mask
        )
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (
            1.0 - boundary_mask
        ) * rotated_base + boundary_mask * rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"BoundlessRotatedSpaceFixedDimensionIntervention()"
    

# Data Batching
def vanilla_collate_fn(tokenizer):
    def inner_collate(batch):
        """
        """
        input_ids, labels = tuple([data_pair[key] for data_pair in batch] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
    return inner_collate
        
def intervention_collate_fn(tokenizer):
    def inner_collate(batch):
        """
        """
        base_input_ids, base_labels, source_input_ids, source_labels, source_predictive_token_idxs, base_predictive_token_idxs = tuple(
            [data_pair[key] for data_pair in batch] for key in 
            ('base_input_ids', 'base_labels', 'source_input_ids', 'source_labels', 'source_predictive_token_idxs', 'base_predictive_token_idxs')
        )
        
        base_input_ids = torch.nn.utils.rnn.pad_sequence(
            base_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        
        source_input_ids = torch.nn.utils.rnn.pad_sequence(
            source_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        
        base_labels = torch.nn.utils.rnn.pad_sequence(base_labels, batch_first=True, padding_value=IGNORE_INDEX)
        source_labels = torch.nn.utils.rnn.pad_sequence(source_labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        source_predictive_token_idxs = torch.LongTensor(source_predictive_token_idxs)
        base_predictive_token_idxs = torch.LongTensor(base_predictive_token_idxs)
        
        return dict(
            base_input_ids=base_input_ids,
            base_labels=base_labels,
            base_attention_mask=base_input_ids.ne(tokenizer.pad_token_id),
            source_input_ids=source_input_ids,
            source_labels=source_labels,
            source_attention_mask=source_input_ids.ne(tokenizer.pad_token_id),
            base_predictive_token_idxs=base_predictive_token_idxs,
            source_predictive_token_idxs=source_predictive_token_idxs
        )
    return inner_collate

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, return_tensors="pt", padding="longest", max_length=1024, truncation=True) for text in strings]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    Preprocess the data by tokenizing.
    """
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]    
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    
    input_ids = [ids[:-1] for ids in input_ids] # remove the last token
    labels = [label[1:] for label in labels]    # remove the first token
    
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len - 1] = IGNORE_INDEX
        
    if len(sources) == len(targets) == 1: 
        return dict(input_ids=input_ids[0], labels=labels[0])
    else:
        return dict(input_ids=input_ids, labels=labels)

def process_dataset(dataset, model_config, tokenizer, n_shots, data_split, prefixes, separators, n_trials=None, ablation_method="zero_shot", draw_source_from_split=True):
    """
    """
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
            base_predictive_token_idxs = token_labels[-1][0]
            data_pair["base_predictive_token_idxs"] = base_predictive_token_idxs
            
        torch_dataset.append(data_pair)
            
    torch_dataset = Dataset.from_list(torch_dataset)
    torch_dataset.set_format(type='torch')
    return torch_dataset

def process_dataloader(dataset, model_config, tokenizer, batch_size, n_shots, data_split, prefixes, separators, collate_fn, n_trials=None, ablation_method="zero_shot", shuffle=False, draw_source_from_split=True):
    """
    """
    torch_dataset = process_dataset(dataset, model_config, tokenizer, n_shots, data_split, prefixes, separators, n_trials, ablation_method, draw_source_from_split)
    torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return torch_dataloader

def process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, data_split, prefixes, separators, collate_fn, n_trials=None, ablation_method="zero_shot", draw_source_from_split=True):
    """
    """
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

def batch_subspace_swap(batch, layer, model:nnsight.LanguageModel, subspace_proj): #, batch_size=16
    """
    Batched subspace_swap intervention at a single layer using nnsight
    """
    batch_size = len(batch['base_input_ids'])
    all_inds = torch.arange(batch_size)
        
    base_prompt, source_prompt = batch['base_input_ids'][:batch_size], batch['source_input_ids'][:batch_size]
    base_intervention_token_idx, source_intervention_token_idx = batch['base_predictive_token_idxs'][:batch_size], batch['source_predictive_token_idxs'][:batch_size]

    with model.trace(validate=False) as tracer:
        with tracer.invoke(base_prompt, scan=False):
            base = model.model.layers[layer].output[0].save()
        
        with tracer.invoke(source_prompt, scan=False):
            source = model.model.layers[layer].output[0].save()
    
    with model.trace(validate=False) as tracer:
        # intervention
        with tracer.invoke(base_prompt, scan=False):
            B = base[all_inds,base_intervention_token_idx,:]
            S = source[all_inds,source_intervention_token_idx,:]

            mixed_out = subspace_proj(B, S, batch_size)
            model.model.layers[layer].output[0][all_inds,base_intervention_token_idx,:] = mixed_out
            del base, source, B,S
        save_out = model.output.save()
    
    
    output_logits = save_out.value.logits
    del save_out
    return output_logits



def batch_subspace_swap_multilayer(batch, layers, model:nnsight.LanguageModel, subspace_proj): #, batch_size=16
    """
    Batched subspace_swap intervention at a single layer using nnsight
    """
    batch_size = len(batch['base_input_ids'])
    all_inds = torch.arange(batch_size)
        
    base_prompt, source_prompt = batch['base_input_ids'][:batch_size], batch['source_input_ids'][:batch_size]
    base_intervention_token_idx, source_intervention_token_idx = batch['base_predictive_token_idxs'][:batch_size], batch['source_predictive_token_idxs'][:batch_size]
    
    bases, sources = [], []

    for layer in layers:
        with model.trace(validate=False) as tracer:
            with tracer.invoke(base_prompt, scan=False):
                base = model.model.layers[layer].output[0].save()
                bases.append(base)
            
            with tracer.invoke(source_prompt, scan=False):
                source = model.model.layers[layer].output[0].save()
                sources.append(source)
    
    with model.trace(validate=False) as tracer:
        # intervention
        with tracer.invoke(base_prompt, scan=False):
            
            for layer, base, source in zip(layers, bases, sources):
                B = base[all_inds,base_intervention_token_idx,:]
                S = source[all_inds,source_intervention_token_idx,:]

                mixed_out = subspace_proj(B, S, batch_size)
                model.model.layers[layer].output[0][all_inds,base_intervention_token_idx,:] = mixed_out
                del base, source, B,S
                
        save_out = model.output.save()
    
    
    output_logits = save_out.value.logits
    del save_out
    return output_logits


def batch_subspace_swap_by_attentions(batch, model:nnsight.LanguageModel, subspace_projs): #, batch_size=16
    """
    Batched subspace_swap intervention at a single layer using nnsight
    """
    batch_size = len(batch['base_input_ids'])
    all_inds = torch.arange(batch_size)
    
    att_head_dim = model.model.config.hidden_size // model.model.config.num_attention_heads

    base_prompt, source_prompt = batch['base_input_ids'][:batch_size], batch['source_input_ids'][:batch_size]
    base_intervention_token_idx, source_intervention_token_idx = batch['base_predictive_token_idxs'][:batch_size], batch['source_predictive_token_idxs'][:batch_size]
    
    bases, sources = [], []
    layers, idxs = [], []
    
    all_layers = sorted(list(subspace_projs.keys()))

    for layer in all_layers:
        for idx in subspace_projs[layer]:
            layers.append(layer)
            idxs.append(idx)
            
            start_dim_idx = idx * att_head_dim
            end_dim_idx = (idx + 1) * att_head_dim
            
            with model.trace(validate=False) as tracer:
                with tracer.invoke(base_prompt, scan=False):
                    base = model.model.layers[layer].self_attn.o_proj.input[0][0][all_inds, :, start_dim_idx:end_dim_idx].save()
                    bases.append(base)
                
                with tracer.invoke(source_prompt, scan=False):
                    source = model.model.layers[layer].self_attn.o_proj.input[0][0][all_inds, :, start_dim_idx:end_dim_idx].save()
                    sources.append(source)                    

    with model.trace(validate=False) as tracer:
        # intervention
        with tracer.invoke(base_prompt, scan=False):
            for layer, idx, base, source in zip(layers, idxs, bases, sources):
                
                subspace_proj = subspace_projs[layer][idx]
                
                B = base[all_inds,base_intervention_token_idx, :]
                S = source[all_inds,source_intervention_token_idx, :]

                mixed_out = subspace_proj(B, S, batch_size)
                model.model.layers[layer].self_attn.o_proj.input[0][0][all_inds, base_intervention_token_idx, start_dim_idx: end_dim_idx] = mixed_out
                del base, source, B,S
                
        save_out = model.output.save()
    
    
    output_logits = save_out.value.logits
    del save_out
    return output_logits



def compute_metrics(eval_preds, eval_labels, generate_output=False):
    """
    """
    total_count = 0
    correct_count = 0
    
    if generate_output:
        outputs = []
        gts = []
        
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        
        for i in range(eval_label.shape[0]):
            label_idxs = eval_label[i].ne(-100).nonzero().squeeze(-1)
                        
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

def evaluate_no_intervention(model, dataloader, device="cuda", corrupt=False, generate_output=False):
    """
    """
    
    with torch.no_grad():
        
        eval_labels = []
        eval_preds = []
        
        for step, inputs in enumerate(tqdm(dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                    
            if corrupt: # eval on corrupted base input sentences (shuffled/zero-shot, etc.)
                outputs = model.forward(
                    input_ids=inputs['base_input_ids'],
                    labels=inputs['base_labels'],
                    attention_mask=inputs['base_attention_mask']
                )
                eval_labels += [inputs['base_labels']]
            else: # eval on clean source inputs
                outputs = model.forward(
                    input_ids=inputs['source_input_ids'],
                    labels=inputs['source_labels'],
                    attention_mask=inputs['source_attention_mask']
                )
                eval_labels += [inputs['source_labels'].detach().cpu()]
                
            eval_preds += [outputs.logits.detach().cpu()]
        
        eval_metrics = compute_metrics(eval_preds, eval_labels, generate_output=generate_output)
        return eval_metrics

def evaluate_w_subspace_intervention(model, subspace_proj, dataloader, intervention_layer, device="cuda", generate_output=False):
    """
    """
    with torch.no_grad():
        
        eval_labels = []
        eval_preds = []
        
        for step, inputs in enumerate(tqdm(dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            outputs = batch_subspace_swap(inputs, intervention_layer, model, subspace_proj)#, batch_size=dataloader.batch_size)
            eval_labels += [inputs['base_labels'].detach().cpu()]
                
            eval_preds += [outputs.detach().cpu()]
        
        eval_metrics = compute_metrics(eval_preds, eval_labels, generate_output=generate_output)
        return eval_metrics
    
    

def evaluate_w_subspace_intervention_multilayer(model, subspace_proj, dataloader, intervention_layers, device="cuda", generate_output=False):
    """
    """
    with torch.no_grad():
        
        eval_labels = []
        eval_preds = []
        
        for step, inputs in enumerate(tqdm(dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            outputs = batch_subspace_swap_multilayer(inputs, intervention_layers, model, subspace_proj)#, batch_size=dataloader.batch_size)
            eval_labels += [inputs['base_labels'].detach().cpu()]
                
            eval_preds += [outputs.detach().cpu()]
        
        eval_metrics = compute_metrics(eval_preds, eval_labels, generate_output=generate_output)
        return eval_metrics
    
def evaluate_w_subspace_intervention_by_attentions(model, subspace_projs, dataloader, device="cuda", generate_output=False):
    with torch.no_grad():
        
        eval_labels = []
        eval_preds = []
        
        for step, inputs in enumerate(tqdm(dataloader)):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            outputs = batch_subspace_swap_by_attentions(inputs, model, subspace_projs)#, batch_size=dataloader.batch_size)
            eval_labels += [inputs['base_labels'].detach().cpu()]
                
            eval_preds += [outputs.detach().cpu()]
        
        eval_metrics = compute_metrics(eval_preds, eval_labels, generate_output=generate_output)
        return eval_metrics
    

def calculate_loss(logits, labels, subspace_proj, mask_weight=1.5, vocab_size=32000):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    
    
    boundary_loss = mask_weight * subspace_proj.intervention_boundaries.sum()
    loss += boundary_loss
    
    return loss

def calculate_loss_by_attentions(logits, labels, subspace_projs, mask_weight=1.5, vocab_size=32000):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    
    
    boundary_loss = 0
    intervention_num = 0
    for layer in subspace_projs.keys():
        for idx in subspace_projs[layer].keys():
            intervention_num += 1
            boundary_loss += mask_weight * subspace_projs[layer][idx].intervention_boundaries.sum()

    loss += boundary_loss / intervention_num
    return loss