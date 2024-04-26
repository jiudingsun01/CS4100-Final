import sys
sys.path.append("../..")

import torch
from tqdm import tqdm, trange

from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from nnsight_DAS_utils import *

from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *
import seaborn as sns
from datasets import Dataset, concatenate_datasets
# from utils.das_utils import *

import argparse


HELD_IN_DATASETS = [f.replace(".json", "") for f in  os.listdir("../dataset_files/abstractive") if f not in 
                    ["antonym.json", "capitalize.json", "present-past.json", 
                     "english-french.json", "singular-plural.json", "country-capital.json", 
                     "ag_news.json", "commonsense_qa.json", "sentiment.json"]]

HELD_OUT_DATASETS = ["antonym", "capitalize", "present-past", 
                     "english-french", "singular-plural"]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_names', help='Name of the datasets to be loaded', type=list, required=False, default=["antonym"])
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='/work/frink/models/flan-llama-7b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default="../antonym-fv-subspace")
    parser.add_argument('--ie_path', help='Path to the AIE', type=str, required=False, default="../results/AIE/ICL/flan-llama-7b/held_in_tasks/held_in_tasks_indirect_effect.pt")
    parser.add_argument('--intervention_path', help='Path to the trained intervention model', type=str, required=False, default="../fv-debug/checkpoints/epoch_5")
    
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over for indirect_effect", type=int, required=False, default=512)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', required=False, default={"input":"\n", "output":"\n\n", "instructions":""})
    
    # parser.add_argument('--prefixes', help='Prompt template prefixes to be used', required=False, default="specific")
    # parser.add_argument('--separators', help='Prompt template separators to be used', required=False, default="specific")
    
    parser.add_argument('--training_method',type=str, required=False, default='noninformative', choices=['noninformative', 'zero_shot', 'both'])
    
    parser.add_argument('--generate_output', help="Whether or not to generate the outputs and keep the answers", type=bool, required=False, default=True)
        
    # Intervention hyperparameters
    parser.add_argument('--batch_size', help='Batch size of inference and training intervention', type=int, required=False, default=32)
    parser.add_argument('--gradient_accumulation_steps', help='Batch size of inference and training intervention', type=int, required=False, default=1)
    parser.add_argument('--epochs', help='Batch size of inference and training intervention', type=int, required=False, default=6)
    parser.add_argument('--dimension_weights', help='weights assign to the dimension loss', type=float, required=False, default=1.5)
    parser.add_argument('--warmup_ratio', help='Batch size of inference and training intervention', type=float, required=False, default=0.1)
    parser.add_argument('--rotate_lr', help='Batch size of inference and training intervention', type=float, required=False, default=1e-3)
    parser.add_argument('--boundary_lr', help='Batch size of inference and training intervention', type=float, required=False, default=1e-2)
    
    parser.add_argument('--temperature_start', help='Batch size of inference and training intervention', type=float, required=False, default=1.0)
    parser.add_argument('--temperature_end', help='Batch size of inference and training intervention', type=float, required=False, default=0.1)
    
    parser.add_argument('--evaluate_per_epoch', help='Whether or not to run and save the results of eval during training', required=False, default=False)
    parser.add_argument('--checkpoint_every_epoch', type=int, default=1)
    
    parser.add_argument('--train', help='Whether train the projection', type=bool, required=False, default=False)
    
    args = parser.parse_args()
    
    dataset_names = args.dataset_names
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = args.save_path_root
    intervention_path = args.intervention_path
    checkpoint_every_epoch = args.checkpoint_every_epoch
    
    ie_path = args.ie_path
    seed = args.seed
    device = args.device

    test_split = float(args.test_split)
    n_shots = args.n_shots
    n_trials = args.n_trials
    
    if args.prefixes == "specific" and args.separators == "specific":
        prefixes = load_task_specific_prefixes_or_separators(dataset_names, root_data_dir="../template_files", is_prefixes=True)
        separators = load_task_specific_prefixes_or_separators(dataset_names, root_data_dir="../template_files", is_prefixes=False)
    else:
        prefixes = load_prefixes_or_separators(args.prefixes)
        separators = load_prefixes_or_separators(args.separators)
    
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    epochs = args.epochs
    warmup_ratio = args.warmup_ratio
    rotate_lr = args.rotate_lr
    boundary_lr = args.boundary_lr
    dimension_weights = args.dimension_weights
    
    temperature_start = args.temperature_start
    temperature_end = args.temperature_end
    
    evaluate_per_epoch = args.evaluate_per_epoch
    training_method = args.training_method
    generate_output = args.generate_output
    
    train = args.train
    
    results = dict()
    
    print(args)
    
    # Load Model & Tokenizer
    print("Loading Model")
    # model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)
    model, tokenizer, model_config = load_nnsight_model(model_name=model_name, device=device)
    # model.to(torch.bfloat16)
    set_requires_grad(model, False)
    
    att_head_dim = model_config["resid_dim"] // model_config["n_heads"]
    top_heads = load_top_k_aie(ie_path, k=10)
    
    intervention_projections = dict()
    if intervention_path is not None:
        all_interventions = [f.split(".")[0] for f in os.listdir(intervention_path)]
        for intervention in all_interventions:
            layer, head_idx = intervention.split("-")
            head_projection = BoundlessRotatedSpaceIntervention(att_head_dim).to('cuda')
            head_projection.load_state_dict(torch.load(f"{intervention_path}/{intervention}.bin"))
            
            if layer not in intervention_projections.keys():
                intervention_projections[layer] = dict()
                
            intervention_projections[layer][head_idx] = head_projection
    else:
        for layer, idx, _ in top_heads:
            head_projection = BoundlessRotatedSpaceIntervention(att_head_dim).to('cuda')
            head_projection.rotate_layer.weight = torch.eye(att_head_dim).to('cuda')
            
            if layer not in intervention_projections.keys():
                intervention_projections[layer] = dict()
            
            intervention_projections[layer][idx] = head_projection
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
                
    # Load the dataset
    print("Loading Dataset")
    set_seed(seed)
    datasets = [load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed) for dataset_name in dataset_names]
    
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
        
    print("Processing Dataloaders")
    eval_no_intervention_dataloader = process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, "valid", prefixes, separators, intervention_collate_fn(tokenizer), ablation_method="zero_shot", draw_source_from_split=True)
    if training_method == "both":
        
        all_datasets = []
        for method in ["zero_shot", "noninformative"]:
            for dataset in datasets:
                all_datasets.append(process_dataset(dataset, model_config, tokenizer, n_shots, "train", prefixes, separators, n_trials=n_trials, ablation_method=method, draw_source_from_split=False))
        train_dataset = concatenate_datasets(all_datasets)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=intervention_collate_fn(tokenizer))
    else:
        train_dataloader = process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, "train", prefixes, separators, intervention_collate_fn(tokenizer), n_trials=n_trials, ablation_method=training_method, draw_source_from_split=False)
        
    fs_eval_dataloader = process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, "valid", prefixes, separators, intervention_collate_fn(tokenizer), ablation_method="noninformative", draw_source_from_split=True)
    zs_eval_dataloader = process_mixed_dataloader(datasets, model_config, tokenizer, batch_size, n_shots, "valid", prefixes, separators, intervention_collate_fn(tokenizer), ablation_method="zero_shot", draw_source_from_split=True)
    

    print(f"Evaluating the model {n_shots}-shots without intervention...")
    eval_dict = evaluate_no_intervention(model, eval_no_intervention_dataloader, device=model.device, generate_output=generate_output)
    print(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: {eval_dict['accuracy']}")
    results['prealign_val_task_accuracy'] = eval_dict["accuracy"]
    
    t_total = int(len(train_dataloader) * epochs)
    warm_up_steps = 0.1 * t_total
    
    target_total_step = len(train_dataloader) * epochs
    
    temperature_schedule = torch.linspace(
        temperature_start, temperature_end, target_total_step
    ).to(torch.bfloat16).to(device)
    
    # Define params to be learned
    optimizer_params = []
    param_count = 0
    total_step = 0
    
    for layer in intervention_projections.keys():
        for idx in intervention_projections[layer].keys():
            optimizer_params += [{'params': intervention_projections[layer][idx].rotate_layer.parameters()}]
            optimizer_params += [{'params': intervention_projections[layer][idx].intervention_boundaries, 'lr': boundary_lr}]
            
            param_count += count_parameters(intervention_projections[layer][idx])
            intervention_projections[layer][idx].set_temperature(temperature_schedule[total_step])
            intervention_projections[layer][idx].train()
            
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=rotate_lr,
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps,
        num_training_steps=t_total
    )
    
    print("subspace_proj trainable parameters: ", param_count)
    print("subspace_proj mask sparsity: ", compute_rotation_mask_sparsity_by_attentions(intervention_projections))
        
    train_iterator = trange(
        0, int(epochs), desc="Epoch"
    )
    
    training_log_dicts = None
    
    os.makedirs(os.path.join(save_path_root, "checkpoints"), exist_ok=True)
    
    if train:
    
        training_log_dicts = []
                
        for epoch in train_iterator:
            
            log_dicts = []
            ckpt_path = os.path.join(save_path_root, "checkpoints", f"epoch_{epoch}")
            
            epoch_iterator = tqdm(
                train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
            )
            
            for step, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
                # b_s = inputs["base_input_ids"].shape[0]

                counterfactual_outputs = batch_subspace_swap_by_attentions(inputs, model, intervention_projections)
                
                eval_metrics = compute_metrics(
                    [counterfactual_outputs], [inputs['base_labels']]
                )
                
                loss = calculate_loss_by_attentions(counterfactual_outputs, inputs["base_labels"], intervention_projections, dimension_weights, model.config.vocab_size)
                loss_str = round(loss.item(), 2)
                
                log_dict = {'loss': loss_str, 'acc': eval_metrics["accuracy"], 'sparsity': compute_rotation_mask_sparsity_by_attentions(intervention_projections)}
                epoch_iterator.set_postfix(log_dict)
                
                log_dicts.append(log_dict)
                
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                    
                loss.backward()
                if total_step % gradient_accumulation_steps == 0:
                    if not (gradient_accumulation_steps > 1 and total_step == 0):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        for layer in intervention_projections.keys():
                            for idx in intervention_projections[layer].keys():
                                intervention_projections[layer][idx].set_temperature(temperature_schedule[total_step])
                        
                total_step += 1
            
            ave_loss = round(sum([log_dict['loss'] for log_dict in log_dicts])/len(log_dicts), 4)
            ave_acc = round(sum([log_dict['acc'] for log_dict in log_dicts])/len(log_dicts), 4)
            ave_sparsity = round((sum([log_dict['sparsity'] for log_dict in log_dicts])/len(log_dicts)).item(), 4) 
            
            epoch_training_log = {'loss': ave_loss, 'acc': ave_acc, 'sparsity': ave_sparsity}
            print("Epoch " + str(epoch) + " finished! Training loss: " + str(ave_loss) + ", training acc: " + str(ave_acc) + ", sparsity: " + str(ave_sparsity))
            
            if evaluate_per_epoch:
                
                fs_shuffled_acc = evaluate_w_subspace_intervention_by_attentions(model, intervention_projections, fs_eval_dataloader, device=model.device, generate_output=generate_output)
                epoch_training_log['fs_shuffled_with_intervention_accuracy'] = fs_shuffled_acc["accuracy"]
                
                zs_intervention_acc = evaluate_w_subspace_intervention_by_attentions(model, intervention_projections, zs_eval_dataloader, device=model.device, generate_output=generate_output)
                epoch_training_log['zs_with_intervention_accuracy'] = zs_intervention_acc["accuracy"]
                
                print("Few-shot shuffled with intervention accuracy: " + str(epoch_training_log['fs_shuffled_with_intervention_accuracy']))
                print("Zero-shot with intervention accuracy: " + str(epoch_training_log['zs_with_intervention_accuracy']))
            
            if epoch % checkpoint_every_epoch == 0:
                # intervenable.save(ckpt_path)
                os.makedirs(ckpt_path, exist_ok=True)
                for layer in intervention_projections.keys():
                    for idx in intervention_projections[layer].keys():
                        torch.save(intervention_projections[layer][idx].state_dict(), f"{ckpt_path}/{layer}-{idx}.bin")
                
            training_log_dicts.append(epoch_training_log)
        
    print("Evaluation the model with intervention...")
    
    fs_shuffled_acc = evaluate_w_subspace_intervention_by_attentions(model, intervention_projections, fs_eval_dataloader, device=model.device, generate_output=generate_output)
    results['fs_shuffled_with_intervention_accuracy'] = fs_shuffled_acc["accuracy"]
    
    fs_shuffled_no_intervention_acc = evaluate_no_intervention(model, fs_eval_dataloader, device=model.device, corrupt=True, generate_output=generate_output)
    results['fs_shuffled_no_intervention_accuracy'] = fs_shuffled_no_intervention_acc["accuracy"]
    
    zs_intervention_acc = evaluate_w_subspace_intervention_by_attentions(model, intervention_projections, zs_eval_dataloader, device=model.device, generate_output=generate_output)
    results['zs_with_intervention_accuracy'] = zs_intervention_acc["accuracy"]
        
    zs_no_intervention_acc = evaluate_no_intervention(model, zs_eval_dataloader, device=model.device, corrupt=True, generate_output=generate_output)
    results['zs_no_intervention_accuracy'] = zs_no_intervention_acc["accuracy"]
    
    print("Few-shot shuffled with intervention accuracy: " + str(results['fs_shuffled_with_intervention_accuracy']))
    print("Few-shot shuffled no intervention accuracy: " + str(results['fs_shuffled_no_intervention_accuracy']))
    print("Zero-shot with intervention accuracy: " + str(results['zs_with_intervention_accuracy']))
    print("Zero-shot no intervention accuracy: " + str(results['zs_no_intervention_accuracy']))
    
    print("Saving results...")
    
    if generate_output:
        
        os.makedirs(os.path.join(save_path_root, "outputs"), exist_ok=True)
        
        with open(f"{save_path_root}/outputs/baseline_outputs.txt", "w") as f:
            for output, label in zip(eval_dict["outputs"], eval_dict["labels"]):
                # f.write(f"{tokenizer.decode(output)}\t{tokenizer.decode(label)}\n")
                f.write(f"Target: {tokenizer.decode(label[0])} ({tokenizer.decode(label)})\nOutput: {tokenizer.decode(output[0])}\n")
            f.close()
        
        with open(f"{save_path_root}/outputs/fs_shuffled_with_intervention_outputs.txt", "w") as f:
            for output, label in zip(fs_shuffled_acc["outputs"], fs_shuffled_acc["labels"]):
                # f.write(f"{tokenizer.decode(output)}\t{tokenizer.decode(label)}\n")
                f.write(f"Target: {tokenizer.decode(label[0])} ({tokenizer.decode(label)})\nOutput: {tokenizer.decode(output[0])}\n")
            f.close()
            
        with open(f"{save_path_root}/outputs/fs_shuffled_no_intervention_outputs.txt", "w") as f:
            for output, label in zip(fs_shuffled_no_intervention_acc["outputs"], fs_shuffled_no_intervention_acc["labels"]):
                # f.write(f"{tokenizer.decode(output)}\t{tokenizer.decode(label)}\n")
                f.write(f"Target: {tokenizer.decode(label[0])} ({tokenizer.decode(label)})\nOutput: {tokenizer.decode(output[0])}\n")
            f.close()
            
        with open(f"{save_path_root}/outputs/zs_with_intervention_outputs.txt", "w") as f:
            for output, label in zip(zs_intervention_acc["outputs"], zs_intervention_acc["labels"]):
                # f.write(f"{tokenizer.decode(output)}\t{tokenizer.decode(label)}\n")
                f.write(f"Target: {tokenizer.decode(label[0])} ({tokenizer.decode(label)})\nOutput: {tokenizer.decode(output[0])}\n")
            f.close()
            
        with open(f"{save_path_root}/outputs/zs_no_intervention_outputs.txt", "w") as f:
            for output, label in zip(zs_no_intervention_acc["outputs"], zs_no_intervention_acc["labels"]):
                # f.write(f"{tokenizer.decode(output)}\t{tokenizer.decode(label)}\n")
                f.write(f"Target: {tokenizer.decode(label[0])} ({tokenizer.decode(label)})\nOutput: {tokenizer.decode(output[0])}\n")
            f.close()
            
    
    with open(f"{save_path_root}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
        f.close()
        
    with open(f"{save_path_root}/results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    if training_log_dicts is not None:
        with open(f"{save_path_root}/training_log.json", "w") as f:
            json.dump(training_log_dicts, f, indent=4)
            f.close()
            
    # intervenable.save(f"{save_path_root}/intervention_model")
    if not os.path.exists(f"{save_path_root}/intervention_model"):
        os.makedirs(f"{save_path_root}/intervention_model")
    
    for layer in intervention_projections.keys():
        for idx in intervention_projections[layer].keys():
            torch.save(intervention_projections[layer][idx].state_dict(), f"{save_path_root}/intervention_model/{layer}-{idx}.bin")
            
    print("Done!")
    