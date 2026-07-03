from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
from peft import PeftModel, PeftConfig, get_peft_model
# from modelscope.msdatasets import MsDataset
import torch
import json
import os
import re
import random
import warnings
warnings.filterwarnings("ignore")
def extract_answer(text):
    pattern = r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        solution_content = match.group(1).strip()
        # print("Extracted content:\n")
        # print(solution_content)
        return solution_content
    else:
        # print("No matching content found.")
        return None
def collate_fn(batch, tokenizer, max_length):
    """
    batch: list of raw text samples (str)
    tokenizer: huggingface tokenizer
    max_length: maximum length to pad to (int)
    """
    encoded_batch = []
    for text in batch:
        # Encode text, return dictionary, note no automatic padding
        enc = tokenizer(text["text"], add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)  # (seq_len,)

        # Add eos_token_id
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer does not have eos_token_id")

        input_ids = torch.cat([input_ids, torch.tensor([eos_id], device=input_ids.device)])

        # Padding to max_length
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer does not have pad_token_id")

        seq_len = input_ids.size(0)
        if seq_len > max_length:
            # Truncate if too long
            input_ids = input_ids[:max_length]
        else:
            # Pad right side if not long enough
            pad_len = max_length - seq_len
            padding = torch.full((pad_len,), pad_id, device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, padding])

        encoded_batch.append(input_ids)

    return torch.stack(encoded_batch)

def prepare_dataloader(data, tokenizer, batch_size, max_length):
    dataset = CustomDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        collate_fn  = lambda x: collate_fn(x, tokenizer, max_length=max_length),
        num_workers = 0,
        shuffle     = True,
        pin_memory  = True,
    )

    return dataloader

def read_math():
    math_data = []
    dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    for item in dataset:
        math_data.append({"question": item['question'], "answer": item['answer']})
    return math_data

def read_python():
    python_data = []
    dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    for item in dataset:
        python_data.append({"question": item['question'], "answer": item['answer']})
    return python_data

def read_numinamath():
    math_data = read_math()
    python_data = read_python()
    return math_data + python_data

def read_bs(config=None):
    data=[]
    # Get path from config, use default path if no config
    if config and hasattr(config, 'paths') and hasattr(config.paths, 'data') and hasattr(config.paths.data, 'bs'):
        dataset_path = config.paths.data.bs
    else:
        dataset_path = "BytedTsinghua-SIA/DAPO-Math-17k"
    
    dataset=load_dataset(dataset_path, split="train[:5000]")
    for item in dataset:
        data.append({"question": item['prompt'][0]["content"], "answer": item['reward_model']["ground_truth"]})
    return data

def read_bs_easy(config=None):
    data=[]
    # Get path from config, use default path if no config
    if config and hasattr(config, 'paths') and hasattr(config.paths, 'data') and hasattr(config.paths.data, 'bs_easy'):
        dataset_path = config.paths.data.bs_easy
    else:
        dataset_path = "Lansechen/bs17k_collection_filtered_easy_maxlength600"
    
    dataset=load_dataset(dataset_path, split="train")
    for item in dataset:
        data.append({"question": item['question'], "answer": item['qwen7b_answer']})
    return data

def read_bs_17k():
    data=[]
    dataset=load_dataset("BytedTsinghua-SIA/DAPO-Math-17k",split="train")
    for item in dataset:
        item=item["conversations"]
        data.append({"question": item[0]['value'], "answer": extract_answer(item[1]['value'])})
    return data
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def read_llada(file_path=None):
    file_path = file_path or os.environ.get("TRADO_LLADA_DATA")
    if not file_path:
        raise ValueError("Set TRADO_LLADA_DATA or pass file_path to read_llada().")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError:
                print(f'JSONDecodeError: {line}')
    return data
def get_bs17k_dataloader(tokenizer, config, max_length=3072):
    train_dataset = []
    # Pass global config to data reading functions
    global_config = getattr(config, '_parent', config)  # Try to get parent config
    data_dict=read_bs(global_config)
    for data in data_dict:
        question = data['question']
        answer = data['answer']

        # messages = [
        #     {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
        # ]
        messages = [
            {"role": "user", "content": question}
        ]
        question = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).input_ids[0]

        # question = tokenizer(question, return_tensors='pt')['input_ids'][0]
        answer = tokenizer(answer, return_tensors='pt')['input_ids'][0]
        answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        question_length = question.shape[-1]
        answer_length = answer.shape[-1]
        combined_length = question_length + answer_length
        if question_length > max_length-100:
            continue 
        if combined_length > max_length:
            padded_data = torch.cat((question, answer), dim=-1)
            padded_data = padded_data[:max_length]  # Truncate to max_length
        else:
            padding_length = max_length - combined_length
            padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
            padded_data = torch.cat((question, answer, padding), dim=-1)

        train_dataset.append(
            dict(
                data = padded_data,
                question_length = question_length,
                length = combined_length,
            )
        )

    dataset = CustomDataset(train_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size  = config.batch_size,
        num_workers = 0,
        shuffle     = True,
        pin_memory  = True,
    )

    return dataloader, None, None

# def get_gsm8k_dataloader(tokenizer, config, max_length=1024):
#     train_dataset = []
#     data_dict = read_numinamath()
#     for data in data_dict:
#         question = data['question']
#         answer = data['answer']

#         question = tokenizer(question, return_tensors='pt')['input_ids'][0]
#         answer = tokenizer(answer, return_tensors='pt')['input_ids'][0]
#         answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

#         question_length = question.shape[-1]
#         answer_length = answer.shape[-1]
#         combined_length = question_length + answer_length

#         if combined_length > max_length:
#             continue

#         padding_length = max_length - combined_length
#         padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
#         padded_data = torch.cat((question, answer, padding), dim=-1)

#         train_dataset.append(
#             dict(
#                 data = padded_data,
#                 question_length = question_length,
#                 length = combined_length,
#             )
#         )

#     dataset = CustomDataset(train_dataset)
#     dataloader = DataLoader(
#         dataset,
#         batch_size  = config.batch_size,
#         collate_fn  = lambda x: collate_fn_pad(x, tokenizer, max_length=max_length),
#         num_workers = 0,
#         shuffle     = True,
#         pin_memory  = True,
#     )

#     return dataloader


# def get_llada_bs17k_dataloader(tokenizer, config, max_length=1024):
#     train_dataset = []
#     # Pass global config to data reading functions
#     global_config = getattr(config, '_parent', config)  # Try to get parent config
#     data_dict = read_bs(global_config)
#     python_dict=read_bs_easy(global_config)
#     data_dict=data_dict+python_dict
#     print("Data length:",len(data_dict))
#     # data_dict = read_llada()
#     for data in data_dict:
#         question = data['question']
#         answer = data['answer']

#         # messages = [
#         #     {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
#         # ]
#         messages = [
#             {"role": "user", "content": question}
#         ]
#         question = tokenizer.apply_chat_template(
#             messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
#         ).input_ids[0]

#         # question = tokenizer(question, return_tensors='pt')['input_ids'][0]
#         answer = tokenizer(answer, return_tensors='pt')['input_ids'][0]
#         answer = torch.cat((answer, torch.tensor([126348])), dim=-1)

#         question_length = question.shape[-1]
#         answer_length = answer.shape[-1]
#         combined_length = question_length + answer_length

#         if combined_length > max_length:
#             continue

#         padding_length = max_length - combined_length
#         padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
#         padded_data = torch.cat((question, answer, padding), dim=-1)

#         train_dataset.append(
#             dict(
#                 data = padded_data,
#                 question_length = question_length,
#                 length = combined_length,
#             )
#         )

#     dataset = CustomDataset(train_dataset)
#     dataloader = DataLoader(
#         dataset,
#         batch_size  = config.batch_size,
#         num_workers = 0,
#         shuffle     = True,
#         pin_memory  = True,
#     )

#     return dataloader
import random
class LladaDataset(Dataset):
    def __init__(
        self,
        data_dict,
        rollout_datas,
        tokenizer,
        max_length=1024,
        max_q_length=2048,
        indices=None,
        train=True,
        random_context_min_length=None,
        random_context_max_length=None,
    ):
        self.data_dict = data_dict
        self.rollout_datas = rollout_datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_q_length = max_q_length
        self.random_context_min_length = random_context_min_length
        self.random_context_max_length = random_context_max_length
        self.eos_token_id = tokenizer.eos_token_id
        
        # If indices are provided, mapping logical idx to real idx
        # If None, use full range
        if indices is None:
            self.indices = list(range(len(data_dict)))
        else:
            self.indices = indices
            
        self.train = train
        
        # Pre-tokenize answers to save time during iteration, but questions (rollouts) 
        # need to be processed dynamically
        self.processed_data = []
        # Optimization: only process data we need? 
        # But data_dict is light, so maybe okay. 
        # Let's simple keep the previous logic but access via self.indices
        
        # We can optimize by only processing the subset, but since we modify logic to use indices map:
        # Pre-process ALL answers for simplicity (or just-in-time, but let's stick to init for now)
        for idx, data in enumerate(data_dict):
             self.processed_data.append({
                 "answer": data["answer"],
                 "question": data['question']
             })

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map to real index in the full dataset
        real_idx = self.indices[idx]
        
        data = self.processed_data[real_idx]
        rollout_text = self.rollout_datas[real_idx % len(self.rollout_datas)]
        
        # 1. Process Rollout / Question
        question_ids = torch.tensor(
            self.tokenizer.encode(rollout_text)
        )
        
        # Dynamic Random Cut logic
        q_len_full = question_ids.shape[0]
        max_context_len = max(1, q_len_full - 100)
        if self.train:
            if self.random_context_min_length is not None and self.random_context_max_length is not None:
                hi = max(1, min(max_context_len, self.random_context_max_length))
                lo = max(1, min(self.random_context_min_length, hi))
                target_len = random.randint(lo, hi) if hi >= lo else hi
            else:
                hi = max(1, min(max_context_len, 8192))
                lo = min(256, hi)
                target_len = random.randint(lo, hi)
        else:
            if self.random_context_max_length is not None:
                target_len = max(1, min(self.random_context_max_length, max_context_len))
            else:
                target_len = max(1, min(5555, max_context_len))
        keep_len = min(q_len_full, target_len)
        question_ids = question_ids[:keep_len]
        
        # Truncate to max_q_length if needed
        if question_ids.shape[0] > self.max_q_length:
             question_ids = question_ids[:self.max_q_length]

        # 2. Process Answer
        answer_ids = self.tokenizer(data["answer"], return_tensors='pt')['input_ids'][0]
        answer_ids = torch.cat((answer_ids, torch.tensor([126348])), dim=-1)

        q_len = question_ids.shape[-1]
        a_len = answer_ids.shape[-1]
        
        # 3. Construct Tensors (No-padding here)
        qa_data = torch.cat((question_ids, answer_ids), dim=-1)
        if qa_data.shape[0] > self.max_length:
            qa_data = qa_data[:self.max_length]
        
        q_only_data = question_ids
        # Truncation already handled above for q_only_data
        
        return dict(
            data=qa_data,
            data_q_only=q_only_data,
            question_length=q_len,
            answer_length=a_len,
        )

def collate_fn_llada(batch, pad_token_id):
    max_qa_len = max(item['data'].size(0) for item in batch)
    max_q_len = max(item['data_q_only'].size(0) for item in batch)
    
    padded_data = []
    padded_q_only = []
    question_lengths = []
    answer_lengths = []
    
    for item in batch:
        # Pad data
        qa = item['data']
        pad_len_qa = max_qa_len - qa.size(0)
        if pad_len_qa > 0:
            qa = torch.cat([qa, torch.full((pad_len_qa,), pad_token_id, dtype=qa.dtype)])
        padded_data.append(qa)
        
        # Pad data_q_only
        q = item['data_q_only']
        pad_len_q = max_q_len - q.size(0)
        if pad_len_q > 0:
            q = torch.cat([q, torch.full((pad_len_q,), pad_token_id, dtype=q.dtype)])
        padded_q_only.append(q)
        
        question_lengths.append(item['question_length'])
        answer_lengths.append(item['answer_length'])
        
    return {
        'data': torch.stack(padded_data),
        'data_q_only': torch.stack(padded_q_only),
        'question_length': torch.tensor(question_lengths),
        'answer_length': torch.tensor(answer_lengths),
    }

def get_llada_bs17k_dataloader(tokenizer, config, max_length=1024, max_q_length=32768):
    train_dataset = []
    global_config = getattr(config, '_parent', config)
    max_length = int(config.get("max_length", max_length))
    max_q_length = int(config.get("max_q_length", max_q_length))
    random_context_min_length = config.get("random_context_min_length", None)
    random_context_max_length = config.get("random_context_max_length", None)
    if random_context_min_length is not None:
        random_context_min_length = int(random_context_min_length)
    if random_context_max_length is not None:
        random_context_max_length = int(random_context_max_length)
    data_dict = read_bs(global_config)
    train_size_limit = config.get("train_size", None)
    if train_size_limit is not None:
        data_dict = data_dict[:int(train_size_limit)]
    
    print("Data length:", len(data_dict))
    
    return_dict=[]
    for idx, data in enumerate(data_dict):
        question = data['question']
        question="Please reason step by step, and put your final answer within \\boxed{}.\n"+question
        messages = [{"role": "user", "content": question}]
        question_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).input_ids[0]
        return_dict.append(tokenizer.decode(question_ids))

    rollout_data_path = None
    if global_config is not None and hasattr(global_config, "paths"):
        rollout_data_path = global_config.paths.get("rollout_data", None)
    rollout_data_path = os.environ.get("TRADO_ROLLOUT_DATA", rollout_data_path)
    if rollout_data_path:
        rollout_data_path = os.path.expanduser(str(rollout_data_path))
        if not os.path.exists(rollout_data_path):
            raise FileNotFoundError(
                f"Training rollout data not found: {rollout_data_path}. "
                "Generate it with scripts/prepare_trajectories.sh or set ROLLOUT_DATA/TRADO_ROLLOUT_DATA."
            )
        rollout_datas = torch.load(rollout_data_path, map_location="cpu")
        if not isinstance(rollout_datas, list):
            raise TypeError(f"Expected rollout data to be a list of strings, got {type(rollout_datas)!r}")
        if train_size_limit is not None:
            rollout_datas = rollout_datas[:int(train_size_limit)]
    else:
        rollout_datas = return_dict

    # Split validation logic manually to pass distinct datasets
    if config.get("split_val", False):
        val_size = int(config.get("val_size", 128))
        if config.get("overfit_same_sample", False):
            train_indices = list(range(len(data_dict)))
            val_indices = train_indices[:val_size]
        else:
            train_size = len(data_dict) - val_size

            # Use fixed seed generator for consistency
            indices = torch.randperm(len(data_dict), generator=torch.Generator().manual_seed(42)).tolist()

            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
        
        # Create separate datasets
        train_ds = LladaDataset(
            data_dict, rollout_datas, tokenizer, max_length, max_q_length,
            indices=train_indices, train=True,
            random_context_min_length=random_context_min_length,
            random_context_max_length=random_context_max_length,
        )
        
        val_ds = LladaDataset(
            data_dict, rollout_datas, tokenizer, max_length, max_q_length,
            indices=val_indices, train=False,
            random_context_min_length=random_context_min_length,
            random_context_max_length=random_context_max_length,
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=lambda b: collate_fn_llada(b, tokenizer.eos_token_id),
            pin_memory=True,
        )
        val_batch_size = config.get("val_batch_size", 4)
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            num_workers=0,
            shuffle=False,
            collate_fn=lambda b: collate_fn_llada(b, tokenizer.eos_token_id),
            pin_memory=True,
        )
        return train_loader, val_loader, return_dict
    
    # If no split, just use training mode
    dataset = LladaDataset(
        data_dict,
        rollout_datas,
        tokenizer,
        max_length,
        max_q_length,
        train=True,
        random_context_min_length=random_context_min_length,
        random_context_max_length=random_context_max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=lambda b: collate_fn_llada(b, tokenizer.eos_token_id),
        pin_memory=True,
    )
    return dataloader, None, return_dict


if __name__ == "__main__":
      text="<|begin_of_thought|>\n\nOkay, let me try to figure out this problem. So, we have this operation defined as a⊗b = a²/b. And we need to compute [(1⊗2)⊗3] - [1⊗(2⊗3)]. Then choose the correct answer from the options given. Alright, let's break it down step by step.\n\nFirst, I need to remember that the operation ⊗ is not associative, right? Because the problem is asking for the difference between two different groupings: (1⊗2)⊗3 and 1⊗(2⊗3). So, the order in which we perform the operations matters here. That's probably why there's a subtraction between them.\n\nLet me start by computing each part separately. Let's tackle the first part: (1⊗2)⊗3.\n\nStarting with the innermost operation, which is 1⊗2. According to the definition, a⊗b = a²/b. So here, a is 1 and b is 2. Plugging those in: 1² / 2 = 1/2. So, 1⊗2 equals 1/2.\n\nNow, we take that result and perform the next operation with 3. So, (1⊗2)⊗3 becomes (1/2)⊗3. Again, using the same definition: a is now 1/2 and b is 3. So, ( (1/2)² ) / 3 = (1/4) / 3 = 1/12. So, (1⊗2)⊗3 equals 1/12.\n\nAlright, that's the first part. Now let's compute the second part: 1⊗(2⊗3). Again, starting with the innermost operation, which is 2⊗3. Applying the definition: a is 2 and b is 3. So, 2² / 3 = 4/3. Therefore, 2⊗3 equals 4/3.\n\nNow, we need to compute 1⊗(4/3). Here, a is 1 and b is 4/3. Using the operation definition: 1² / (4/3) = 1 / (4/3) = 3/4. So, 1⊗(2⊗3) equals 3/4.\n\nNow, the problem asks for the difference between the two results: [(1⊗2)⊗3] - [1⊗(2⊗3)] = (1/12) - (3/4). To subtract these fractions, they need a common denominator. The denominators are 12 and 4, so 12 is the common denominator.\n\nConverting 3/4 to twelfths: 3/4 = 9/12. So, 1/12 - 9/12 = (1 - 9)/12 = -8/12. Simplifying that fraction by dividing numerator and denominator by 4: -8/12 = -2/3.\n\nHmm, looking at the answer choices, option A is -2/3. So, is that the answer? Wait, but let me double-check my calculations to make sure I didn't make a mistake somewhere.\n\nFirst, checking (1⊗2): 1² / 2 = 1/2. Correct. Then, (1/2)⊗3: (1/2)² / 3 = (1/4)/3 = 1/12. That seems right.\n\nNow, for 2⊗3: 2² / 3 = 4/3. Correct. Then, 1⊗(4/3): 1² / (4/3) = 1 / (4/3) = 3/4. Yes, that's correct.\n\nSubtracting 3/4 from 1/12: 1/12 - 3/4. Convert 3/4 to 9/12, so 1/12 - 9/12 = -8/12 = -2/3. Yes, that all checks out. So the answer should be -2/3, which is option A.\n\nWait, but let me think again. The operation is defined for all nonzero numbers, so we don't have any issues with division by zero here. 2⊗3 is 4/3, which is fine, and then 1⊗(4/3) is 3/4. Correct.\n\nAlternatively, maybe there's a different way to approach the problem? Let me try expanding both expressions using variables to see if there's a pattern.\n\nLet's denote the first expression: (a⊗b)⊗c. Using the definition:\n\nFirst, compute a⊗b = a²/b.\n\nThen, take that result and ⊗ with c: (a²/b)⊗c = ( (a²/b)² ) / c = a⁴ / (b² c).\n\nNow, the second expression: a⊗(b⊗c). First compute b⊗c = b²/c.\n\nThen, a⊗(b²/c) = a² / (b²/c) = a² * (c / b²) = (a² c) / b².\n\nTherefore, the difference between the two expressions is:\n\n(a⁴ / (b² c)) - (a² c / b²) = (a⁴ - a² c²) / (b² c) = a² (a² - c²) / (b² c).\n\nHmm, factoring that, it's a² (a - c)(a + c) / (b² c).\n\nBut in our specific problem, a = 1, b = 2, c = 3. Plugging those values in:\n\n1² (1 - 3)(1 + 3) / (2² * 3) = 1 * (-2)(4) / (4 * 3) = (-8) / 12 = -2/3. Same result. So that confirms the answer is indeed -2/3.\n\nTherefore, I think my initial calculation was correct, and the answer is option A.\n\n**Final Answer**\n\\boxed{A}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\nTo determine the value of \\([(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)]\\) where the operation \\(\\otimes\\) is defined by \\(a \\otimes b = \\frac{a^2}{b}\\), we proceed as follows:\n\nFirst, compute \\(1 \\otimes 2\\):\n\\[\n1 \\otimes 2 = \\frac{1^2}{2} = \\frac{1}{2}\n\\]\nNext, use this result to compute \\((1 \\otimes 2) \\otimes 3\\):\n\\[\n\\left(\\frac{1}{2}\\right) \\otimes 3 = \\frac{\\left(\\frac{1}{2}\\right)^2}{3} = \\frac{\\frac{1}{4}}{3} = \\frac{1}{12}\n\\]\n\nNow, compute \\(2 \\otimes 3\\):\n\\[\n2 \\otimes 3 = \\frac{2^2}{3} = \\frac{4}{3}\n\\]\nThen, use this result to compute \\(1 \\otimes (2 \\otimes 3)\\):\n\\[\n1 \\otimes \\left(\\frac{4}{3}\\right) = \\frac{1^2}{\\frac{4}{3}} = \\frac{1}{\\frac{4}{3}} = \\frac{3}{4}\n\\]\n\nFinally, find the difference between the two results:\n\\[\n\\frac{1}{12} - \\frac{3}{4} = \\frac{1}{12} - \\frac{9}{12} = \\frac{1 - 9}{12} = \\frac{-8}{12} = -\\frac{2}{3}\n\\]\n\nThus, the answer is \\(\\boxed{A}\\).\n\n<|end_of_solution|>"
      print(extract_answer(text))

def get_dataloader_by_config(tokenizer, config, global_config=None, max_length=3072):
    """Select different data loaders based on config file"""
    if global_config is None:
        global_config = config
    
    training_mode = global_config.get('training_mode', 'dream')
    
    # Add reference to global config for data loading functions to access
    config._parent = global_config
    
    if training_mode == 'llada':
        return get_llada_bs17k_dataloader(tokenizer, config, max_length)
    elif training_mode == 'dream':
        train_loader, val_loader, return_dict = get_bs17k_dataloader(tokenizer, config, max_length)
        return train_loader, val_loader, return_dict
    elif training_mode == 'trado':
        return get_llada_bs17k_dataloader(tokenizer, config, max_length)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")
