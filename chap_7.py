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
