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

## 5.2 Training an LLM (Pretraining on unlabeled data)
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # reset loss gradient from previous iteration
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward() # calculate loss gradients
            optimizer.step() # update model weights using loss gradient
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0: # optional evaluation step
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss: {train_loss:.3f}, "
                      f"Val loss: {val_loss:.3f}")
            
        # print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

# function for printing numeric training and validaiton set losses after each model update
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # dropout is disabled for stable and reproducible results during evaluation
    with torch.no_grad(): # disable gradient tracking, not required during evaluation, and reduce overhead
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()

    return train_loss, val_loss

# function to track model improvement during training
# provides a text example generated by the model to understand its current capabilities
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# train a GPTModel for 10 epochs and AdamW optimizer
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), # returns all trainable weights parameters of the model
    lr=0.0004, weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Evrey effort moves you", tokenizer=tokenizer
)

# plot showing training and validation losses side by side
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
