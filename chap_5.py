## 5.1 Evaluating generative text models
import torch
from chap_4 import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, # lowered comapare to chap.4 (was 1024) to test with laptops
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1, # other common value is 0
    "qkv_bias": False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

import tiktoken
from chap_4 import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # adds the batch dimension

    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # removes batch dimension
    
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

## 5.1.2. Calculating the text generation loss
# input examples mapped to token ids
inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                       [40,    1107, 588]]) # "I really like"]
# target IDs we want the model to compute given the inputs
# (similar to inputs, but shifted one position forward, since we want to predict the "next word")
targets = torch.tensor([[3626, 6100, 345   ],  # [" effort moves you",
                        [1107, 588,  11311]])  # " really like chocolate"]

with torch.no_grad(): # disables gradient tracking since we are not training yet
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1) # probability scores
print(probas.shape) # (n_inputs or batch_size, n_tokens, emb_dim or vocab_size)

# obtain predicted token IDs
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

# convert token IDs into text
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
      f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

# apply the logarithm to the probability scores
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
# combien them in a single score by averaging
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
# calculate the negative average log probability (aka "cross entropy")
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)
# IN DEEP LEARNING THE GOAL IS TO GET THE CROSS ENTROPY LOSS AS CLOSE TO 0 AS POSSIBLE 
# BY UPDATING THE MODEL'S WEIGHTS IN THE TRAINING PROCESS

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss) # same as neg_avg_log_probas, meaning cross_entropy does each step (softmax, log, avg, and neg) in one go

# Perplexity: another loss function
perplexity = torch.exp(loss)
print(perplexity) # e.g. a value of 48 725 means that the model is usure about which among 48 725 to chose in the vocabulary as next token (which is bad)

## 5.1.3 Calculating the training and validation set losses
# use "The Verdict" to compute the loss on the training and validation datasets
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# split data into train and validation
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# import the data loader from chapter 2
from chap_2 import create_data_loader_v1
torch.manual_seed(123)

train_loader = create_data_loader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_data_loader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# check if the data were created correctly
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

# utility function to calculate the cross entropy loss returned by the loader for a single batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device) # to transfer data to a GPU if available
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss

# function using calc_loss_batch to calculate the loss over all the batches
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item() # sums loss for each batch
        else:
            break
    
    return total_loss / num_batches # average the loss over all batches

# apply the loss function to the training an validation loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # use CUDA if available, or CPU otherwise
with torch.no_grad(): # disable gradient since we are not trining yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

## 5.2 Training an LLM
