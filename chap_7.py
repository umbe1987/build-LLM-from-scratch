## 7 Fine-tuning to follow instructions
## 7.2 Preparing a dataset for supervised instruction fine-tuning

# The dataset consists of 1,100 instruction–response pairs
# EXAMPLE
# instruction: Convert 45 kilometers to meters >>>> response: 45 kilometers is 45000 meters
import json
import os
import urllib.request # CHANGED FROM BOOK "import urllib" (see https://stackoverflow.com/questions/37042152/python-3-5-1-urllib-has-no-attribute-request)

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        data = json.load(file)
    
    return data

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))

print("Example entry:\n", data[50])
print("Another example entry:\n", data[999])

# We need to convert the python dictionary format into the prompt style formatting needed for fine-tuning (called "Alpaca-style")
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )

    return instruction_text + input_text

# test the format_input function with one example entry
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)

# testing with an empty input
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)

# divide the dataset into training (85%), validation (5%), and test (10%) sets
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

## 7.3 Organizing data into training batches
# The first 2 of 5 sub-steps of creating batches are
#   1. turn python dicts into prompt style formattings
#   2. tokenize them
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        # pre-tokenizes texts
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
    
# to collect all training examples into batches, they need to have the same length
# so we pad them ("collate") with the (encoded) <|endoftext|> token
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# we define a custom collate function that can be used in the data loader.
# this function pads each batch to the longest entry, while every batch can
# have a different length.
# Example, if the first batch the longest entry is 5 tokens long, all other
# entries are padded so their lenght is 5. The second batch may have
# the longest entry with 4 tokens, so its length will be 4, not 5.
def custom_collate_draft_1(
        batch,
        pad_token_id=50256,     # this is the encoded <|endoftext|> token
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch) # find the longest sequence in the batch
    inputs_lst = []

    for item in batch:      # pads and prepares inputs
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) # remove extra padded token added earlier
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)

    return inputs_tensor

# let's test the function before using it in the data loader
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)
print(custom_collate_draft_1(batch))

# this udpdated collate function also generates the target token IDs for the input token IDs
# These are the inputs shifted right by one position and padded with the <|endoftext|> token
def custom_collate_draft_2(
        batch,
        pad_token_id=50256,
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # shifts +1 to the right for the targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

# let's test the updated collate function
inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)

# last, in this updated collate function the end-of-text padding tokens in 
# the target lists are replaced the placeholder value of -100, so that 
# they don't contribute to the calculuation of the training loss. 
# We only retain the first end-of-text token.
def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,          # IMPORTANT: -100 value is ignored by PyTorch cross entropy loss function!!!
        allowed_max_length=None,    # added in case we need to define a maximum length when using a custom dataset
        device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # replaces add but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # optionally truncates to the maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

# let's try this new collate function to see if it works
inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)

## 7.4 Creating data loaders for an instruction dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# use partial to create a new version of customized_collate_fn that 
# has the required device and maximum length defined as in GPT-2
from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

# setup the data loader with our custom collate function for the batching process
from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

## 7.5 Loading a pretrained LLM
# Download and load the pretrained GPT model (medium size, 355M params)
from gpt_download import download_and_load_gpt2
from chap_4 import GPTModel
from chap_5 import load_weights_into_gpt

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# test the pretrained model without fine-tuning
torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)

# generate the model reponse based on the above ample instruction
from chap_5 import generate, text_to_token_ids, token_ids_to_text

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256
)
generated_text = token_ids_to_text(token_ids, tokenizer)

# since we are evaluating the generated text and not the text 
# completion including the input text, we remove the latter from 
# the output.
response_text = generated_text[len(input_text):].strip()
print(response_text)

## 7.6 Fine-tuning the LLM on instruction data
