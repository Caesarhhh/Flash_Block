import transformers
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig,get_peft_model
import torch
from models.sdar.modeling_sdar import SDARForCausalLM
from models.sdar.configuration_sdar import SDARConfig

def get_model_by_config(config,train=True,device_map="cuda"):
    """Select different models based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'llada':
        return get_llada(config)
    elif training_mode == 'dream':
        return get_model(config)
    elif training_mode == 'trado':
        return get_llada(config,train=train,device_map=device_map)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

def get_model(config):
    # Use path from config, use default path if no config
    model_path = config.paths.model
    
    model = SDARForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
        )
    # print(model.named_modules())
    # print(model,"model
    for param in model.parameters():
        param.requires_grad = False
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "o_proj"],)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer

def get_llada(config,train=True,device_map="cuda"):
    # Use path from config, use default path if no config
    model_path = config.paths.model
    
    model = SDARForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device_map
        )
    # print(model.named_modules())
    # print(model,"model
    # print(model)
    # exit()
    for param in model.parameters():
        param.requires_grad = False
    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if train:
        peft_config = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.1,target_modules=["q_proj", "v_proj","k_proj", "o_proj", "attn_out"],)
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
    return model, tokenizer
# def create_attention_mask(input_ids, mask_id):
#     """
#     Create an attention mask based on the input_ids and mask_id.

#     Args:
#         input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
#         mask_id (int): The ID of the mask token.

#     Returns:
#         torch.Tensor: The attention mask of shape (batch_size, sequence_length, sequence_length).
