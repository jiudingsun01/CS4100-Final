from datasets import Dataset, concatenate_datasets

import sys
sys.path.append("../..")

import re
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm, trange
import pathlib

from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from pyvene.models.configuration_intervenable_model import IntervenableRepresentationConfig, IntervenableConfig
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.interventions import BoundlessRotatedSpaceIntervention
from pyvene.models.basic_utils import set_seed, count_parameters


from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *
from utils.das_utils import *

import argparse


model, tokenizer, model_config = load_gpt_model_and_tokenizer("/work/frink/models/alpaca-7b/", device="cuda")


def intervention_collate_fn(batch):
    base_input_ids, base_labels, source_input_ids, source_labels, source_predictive_token_idxs, predictive_token_idxs = tuple(
        [data_pair[key] for data_pair in batch] for key in 
        ('base_input_ids', 'base_labels', 'source_input_ids', 'source_labels', 'source_predictive_token_idxs', 'predictive_token_idxs')
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
    predictive_token_idxs = torch.LongTensor(predictive_token_idxs)
    
    return dict(
        base_input_ids=base_input_ids,
        base_labels=base_labels,
        base_attention_mask=base_input_ids.ne(tokenizer.pad_token_id),
        source_input_ids=source_input_ids,
        source_labels=source_labels,
        source_attention_mask=source_input_ids.ne(tokenizer.pad_token_id),
        predictive_token_idxs=predictive_token_idxs,
        source_predictive_token_idxs=source_predictive_token_idxs
    )
    
    
data_idxs = [15954, 13665, 2137, 18304, 15949, 5294, 10969, 13788]
alpaca_data = json.load(open("../alpaca_data.json"))

alpaca_data = json.load(open("../alpaca_data.json"))
alpaca_data = [alpaca_data[i] for i in data_idxs]



return_dict = {}
source_inputs, source_labels = [], []
for input_dict in alpaca_data:
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
for input_dict in alpaca_data:
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


train_dataset = Dataset.from_dict(return_dict)
train_dataset.set_format(type='torch')
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=intervention_collate_fn)

intervenable_config = simple_boundless_das_position_config(type(model), "block_output", 12)
intervenable = IntervenableModel(intervenable_config, model)

intervenable.set_device("cuda")
intervenable.disable_model_gradients()

def calculate_loss(logits, labels):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, intervenable.model_config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    
    for k, v in intervenable.interventions.items():
        boundary_loss = 1 * v[0].intervention_boundaries.sum()
    loss += boundary_loss
    
    return loss

for batch in train_dataloader:
    print(batch.keys())
    print(batch["base_input_ids"].shape)
    print(batch["base_labels"].shape)
    print(batch["base_attention_mask"].shape)
    print(batch["predictive_token_idxs"])
    
    print(batch["source_input_ids"].shape)
    print(batch["source_labels"].shape)
    print(batch["source_attention_mask"].shape)
    print(batch["source_predictive_token_idxs"])
    print()
    
    # outputs = model(input_ids=batch["base_input_ids"].cuda(), labels=batch["base_labels"].cuda(), attention_mask=batch["base_attention_mask"].cuda())
    outputs = model(input_ids=batch["source_input_ids"].cuda(), labels=batch["source_labels"].cuda(), attention_mask=batch["source_attention_mask"].cuda())
    break


t_total = int(len(train_dataloader) * 10)
warm_up_steps = 0.1 * t_total
optimizer_params = []

for k, v in intervenable.interventions.items():
    optimizer_params += [{'params': v[0].rotate_layer.parameters()}]
    optimizer_params += [{'params': v[0].intervention_boundaries, 'lr': 1e-2}]

optimizer = torch.optim.Adam(
    optimizer_params,
    lr=1e-3,
)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warm_up_steps,
    num_training_steps=t_total
)

target_total_step = len(train_dataloader) * 10

temperature_schedule = torch.linspace(
    50, 0.1, target_total_step
).to(torch.bfloat16).to("cuda")

total_step = 0
intervenable.set_temperature(temperature_schedule[total_step])

intervenable.model.train() # train enables drop-off but no grads
print("llama trainable parameters: ", count_parameters(intervenable.model))
print("intervention trainable parameters: ", intervenable.count_parameters())

train_iterator = trange(
    0, int(10), desc="Epoch"
)

training_log_dicts = None

    
    
training_log_dicts = []
        
for epoch in train_iterator:
    
    log_dicts = []
    
    epoch_iterator = tqdm(
        train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
    )
    
    for step, inputs in enumerate(epoch_iterator):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")
        b_s = inputs["base_input_ids"].shape[0]

        source2base = ([[[idx] for idx in inputs["source_predictive_token_idxs"].tolist()]], [[[idx] for idx in inputs["predictive_token_idxs"].tolist()]])
        
        all_source_seqs = [tokenizer.decode(inputs["source_input_ids"][i]) for i in range(b_s)]
        _, counterfactual_outputs = intervenable(
            {"input_ids": inputs["base_input_ids"], "attention_mask": inputs["base_attention_mask"]},
            [{"input_ids": inputs["source_input_ids"], "attention_mask": inputs["source_attention_mask"]}],
            {"sources->base": source2base}
        )
        
        eval_metrics = compute_metrics(
            [counterfactual_outputs.logits], [inputs['base_labels']]
        )
        
        loss = calculate_loss(
            counterfactual_outputs.logits, inputs["base_labels"]
        )
        loss_str = round(loss.item(), 2)
        
        log_dict = {'loss': loss_str, 'acc': eval_metrics["accuracy"], 'sparsity': compute_rotation_mask_sparsity(intervenable)}
        epoch_iterator.set_postfix(log_dict)
        
        log_dicts.append(log_dict)
                    
        loss.backward()
        if total_step % 1 == 0:
            optimizer.step()
            scheduler.step()
            intervenable.set_zero_grad()
            intervenable.set_temperature(temperature_schedule[total_step])
            
        total_step += 1
    
    ave_loss = round(sum([log_dict['loss'] for log_dict in log_dicts])/len(log_dicts), 4)
    ave_acc = round(sum([log_dict['acc'] for log_dict in log_dicts])/len(log_dicts), 4)
    ave_sparsity = round(sum([log_dict['sparsity'] for log_dict in log_dicts])/len(log_dicts), 4) 
    
    epoch_training_log = {'loss': ave_loss, 'acc': ave_acc, 'sparsity': ave_sparsity}
    print("Epoch " + str(epoch) + " finished! Training loss: " + str(ave_loss) + ", training acc: " + str(ave_acc) + ", sparsity: " + str(ave_sparsity))
    training_log_dicts.append(epoch_training_log)