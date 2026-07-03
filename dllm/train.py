# from peft import PeftModel, PeftConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import PeftModel, PeftConfig, get_peft_model
from utils.util import flatten_dict,shift_logits
from utils.data import get_bs17k_dataloader,get_llada_bs17k_dataloader,get_dataloader_by_config
from utils.model import get_model,get_llada,get_model_by_config
from utils.loss import compute_loss,compute_llada_loss,compute_normal_loss,compute_loss_by_config
from utils.generation import sample_tokens
# import dataloader
import copy
import os
import torch
import argparse
import torch.distributed as dist
from omegaconf import OmegaConf
import shutil
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import os
from transformers import AutoModelForCausalLM
from peft import PeftModel

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 保证每次结果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_merged_model(model, output_dir,config):
    """
    在 CPU 上重新加载基础模型 + LoRA 并 merge，不占用 GPU。
    """
    cpu_model, tokenizer = get_model_by_config(config,train=True,device_map="cpu")
    cpu_model.load_state_dict(model.state_dict())

    merged = cpu_model.merge_and_unload()

    merged.save_pretrained(output_dir)
    print(f"Saved merged model to: {output_dir}")

    del merged


def copy_non_weight_files(src_dir, dst_dir):
    """
    将 src_dir 中所有非权重文件复制到 dst_dir。
    权重文件：*.bin / *.safetensors / *.pt
    """

    weight_exts = {".bin", ".safetensors", ".pt"}

    os.makedirs(dst_dir, exist_ok=True)

    for root, dirs, files in os.walk(src_dir):
        # 计算在目标目录的相对路径
        rel_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, rel_path)
        os.makedirs(target_root, exist_ok=True)

        for f in files:
            # 跳过权重文件
            if any(f.endswith(ext) for ext in weight_exts):
                continue

            src_file = os.path.join(root, f)
            dst_file = os.path.join(target_root, f)

            # 文件存在则不覆盖
            if os.path.exists(dst_file):
                continue

            shutil.copy2(src_file, dst_file)
            # print(f"Copied: {src_file} -> {dst_file}")

def get_accelerator(config, global_config):
    # Select experiment path based on config
    if hasattr(global_config, 'paths') and hasattr(global_config.paths, 'experiment'):
        root_path = global_config.paths.experiment
    else:
        root_path = config.root if hasattr(config, 'root') else os.environ.get('TRADO_EXPERIMENT_ROOT', 'outputs/experiment')
    
    output_dir = os.path.join(root_path, config.exp_name, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with=None if config.report_to == 'no' else config.report_to,
        mixed_precision=config.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    return accelerator, output_dir

def main(args):
    config = OmegaConf.load(args.config)
    accelerator, output_dir = get_accelerator(config.train, config)
    
    # Use unified model and data loading functions
    teacher_denoiser, tokenizer = get_model_by_config(config,train=False)
    # teacher_denoiser=None
    denoiser, tokenizer = get_model_by_config(config)
    dataloader, val_dataloader, _ = get_dataloader_by_config(tokenizer, config.data, config)
    # dataloader = val_dataloader
    
    if config.train.decoder_resume_path is not None:
        ckpt = torch.load(config.train.decoder_resume_path, map_location='cpu', weights_only=True)
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = denoiser.load_state_dict(ckpt, strict=False)
        if accelerator.is_main_process:
            print(f'model ckpt loaded from {config.train.decoder_resume_path}. Len {len(ckpt)} - Missing {m} - Unexpected {u}.')

        # ckpt = torch.load(config.train.head_resume_path, map_location='cpu', weights_only=True)
        # if config.train.skipped_keys:
        #     ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        # m, u = denoiser.lm_head.load_state_dict(ckpt, strict=False)
        # if accelerator.is_main_process:
        #     print(f'model ckpt loaded from {config.train.head_resume_path}')

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(param for param in denoiser.parameters() if param.requires_grad)
    debug_grad_stats = bool(config.train.get("debug_grad_stats", False))
    debug_grad_steps = int(config.train.get("debug_grad_steps", 5))
    debug_param = next((p for p in params_to_learn if p.requires_grad), None)
    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = config.train.get("adam_eps", 1e-3),
    )
    # optimizer = torch.optim.SGD(
    #     params_to_learn,
    #     lr=config.train.lr,
    #     momentum=0.95,
    #     nesterov=True
    # )

    
    teacher_denoiser, denoiser, dataloader, optimizer = accelerator.prepare(
        teacher_denoiser, denoiser, dataloader, optimizer
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    config.device_count = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))

    training_done = False
    epoch = 0
    progress_bar = tqdm(
        total   = config.train.num_iters,
        initial = global_step,
        desc    = 'Steps',
        disable = not accelerator.is_local_main_process,
    )

    if accelerator.is_main_process:
        print(f'Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e9} B')

    if global_step >= config.train.num_iters:
        progress_bar.close()
        accelerator.end_training()
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    while not training_done:
        if accelerator.is_main_process:
            print(f'Epoch: {epoch}')
        for batch in dataloader:
            with accelerator.accumulate([denoiser]):
                denoiser.train()
                input_ids = batch['data_q_only']
                # print("input_ids",input_ids.dtype)
                question_length = batch['question_length']
                
                # Use unified loss function selection
                losses = compute_loss_by_config(
                    input_ids,
                    teacher_denoiser,
                    denoiser,
                    question_length,
                    block_size    = config.train.block_size,
                    # mask_id       = config.denoiser.encoder.mask_id,
                    mask_id       = 151669,
                    enable_shift  = config.train.enable_shift,
                    share_steps   = config.train.share_steps,
                    self_align    = config.train.self_align,
                    feature_align = config.train.feature_align,
                    self_step     = config.train.self_step,
                    eos_id        = tokenizer.eos_token_id,
                    config        = config,
                    tokenizer     = tokenizer
                )
                
                if config.train.share_steps > 1:
                    loss_tgt = losses['loss']
                    # loss_1 = losses['loss_1']
                    # loss_2 = losses['loss_2']
                else:
                    raise NotImplementedError
                torch.cuda.empty_cache()
                if debug_grad_stats and accelerator.is_main_process and debug_param is not None and global_step < debug_grad_steps:
                    debug_param_before = debug_param.detach().float().norm().item()
                else:
                    debug_param_before = None
                accelerator.backward(loss_tgt)
                if debug_grad_stats and accelerator.is_main_process and global_step < debug_grad_steps:
                    total_grad_sq = 0.0
                    max_grad = 0.0
                    grad_param_count = 0
                    for p in params_to_learn:
                        if p.grad is None:
                            continue
                        grad = p.grad.detach().float()
                        total_grad_sq += grad.norm().item() ** 2
                        max_grad = max(max_grad, grad.abs().max().item())
                        grad_param_count += 1
                    print(
                        f"[grad-debug pre-step={global_step}] "
                        f"loss_requires_grad={loss_tgt.requires_grad} "
                        f"grad_params={grad_param_count}/{len(params_to_learn)} "
                        f"grad_norm={total_grad_sq ** 0.5:.6e} "
                        f"max_grad={max_grad:.6e} "
                        f"param0_norm_before={debug_param_before}"
                    )
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)

                optimizer.step()
                if debug_grad_stats and accelerator.is_main_process and debug_param is not None and global_step < debug_grad_steps:
                    debug_param_after = debug_param.detach().float().norm().item()
                    print(
                        f"[grad-debug post-step={global_step}] "
                        f"param0_norm_after={debug_param_after} "
                        f"delta_norm={abs(debug_param_after - debug_param_before):.6e}"
                    )
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                logs = dict()
                loss_tgt = accelerator.gather(loss_tgt.detach()).mean().item()
                logs['loss'] = loss_tgt
                logs["acc"] = losses["acc"]
                logs["acc_dense"] = losses["acc_dense"]
                logs["kl_loss"] = losses["kl_loss"]
                logs["kl_loss_dense"] = losses["kl_loss_dense"]
                # if config.train.share_steps > 1:
                #     loss_1 = accelerator.gather(loss_1.detach()).mean().item()
                #     loss_2 = accelerator.gather(loss_2.detach()).mean().item()
                    # logs['loss_1'] = loss_1
                    # logs['loss_2'] = loss_2
                    
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

                if global_step > 0 and global_step % config.train.eval_every == 0:
                    if val_dataloader is not None:
                        denoiser.eval()
                        val_losses = []
                        val_accs = []
                        val_accs_dense = []
                        val_kl_losses = []
                        val_kl_losses_dense = []
                        
                        with torch.no_grad():
                            max_val_batches = config.train.get("max_val_batches", None)
                            for step_in_val, val_batch in enumerate(tqdm(val_dataloader, desc='Evaluating', disable=not accelerator.is_local_main_process)):
                                if max_val_batches is not None and step_in_val >= int(max_val_batches):
                                    break
                                val_input_ids = val_batch['data_q_only']
                                val_question_length = val_batch['question_length']
                                
                                
                                val_step_losses = compute_loss_by_config(
                                    val_input_ids,
                                    teacher_denoiser,
                                    denoiser,
                                    val_question_length,
                                    block_size    = config.train.block_size,
                                    mask_id       = config.denoiser.encoder.mask_id,
                                    enable_shift  = config.train.enable_shift,
                                    share_steps   = config.train.share_steps,
                                    self_align    = config.train.self_align,
                                    feature_align = config.train.feature_align,
                                    self_step     = config.train.self_step,
                                    eos_id        = tokenizer.eos_token_id,
                                    config        = config,
                                    tokenizer     = tokenizer
                                )
                                
                                val_losses.append(val_step_losses['loss'].detach())
                                val_accs.append(torch.tensor(val_step_losses['acc'], device=accelerator.device))
                                val_accs_dense.append(torch.tensor(val_step_losses['acc_dense'], device=accelerator.device))
                                val_kl_losses.append(torch.tensor(val_step_losses['kl_loss'], device=accelerator.device))
                                val_kl_losses_dense.append(torch.tensor(val_step_losses['kl_loss_dense'], device=accelerator.device))
                        
                        # Gather and average metrics
                        avg_val_loss = accelerator.gather(torch.stack(val_losses)).mean().item()
                        avg_val_acc = accelerator.gather(torch.stack(val_accs)).mean().item()
                        avg_val_acc_dense = accelerator.gather(torch.stack(val_accs_dense)).mean().item()
                        avg_val_kl_loss = accelerator.gather(torch.stack(val_kl_losses)).mean().item()
                        avg_val_kl_loss_dense = accelerator.gather(torch.stack(val_kl_losses_dense)).mean().item()
                        
                        if accelerator.is_main_process:
                            print(f"\nStep {global_step} Validation: loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}, acc_dense={avg_val_acc_dense:.4f}, kl_loss={avg_val_kl_loss:.4f}, kl_loss_dense={avg_val_kl_loss_dense:.4f}")

                accelerator.wait_for_everyone()

                if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                    denoiser.eval()
                    lora_state_dict={k:v for k,v in denoiser.state_dict().items() if "lora" in k}
                    save_path=os.path.join(output_dir, f"Decoder-{config.train.exp_name}-step{global_step}")
                    os.makedirs(save_path,exist_ok=True)
                    torch.save(lora_state_dict,f"{save_path}/lora.pt")
                    copy_non_weight_files(config.paths.model,save_path)
                    save_merged_model(accelerator.unwrap_model(denoiser),save_path,config)
                    # decoder_state_dict = copy.deepcopy(accelerator.unwrap_model(denoiser)).merge_and_unload().save_pretrained(save_path)
                    # lmhead_state_dict = accelerator.unwrap_model(denoiser).lm_head.state_dict()
                    # torch.save(lmhead_state_dict, os.path.join(output_dir, f"LMhead-{config.train.exp_name}-{global_step // 1000}k"))
                accelerator.wait_for_everyone()
            if global_step >= config.train.num_iters:
                training_done = True
                break
        epoch += 1
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/dream.yaml')
    args = parser.parse_args()
    set_seed(42)
    main(args)    
