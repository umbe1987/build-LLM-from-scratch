## 3.3.1 A simple self-attention mechanism without trainable weights
import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66],  # journey  (x^1)
    [0.57, 0.85, 0.64],  # starts   (x^1)
    [0.22, 0.58, 0.33],  # with     (x^1)
    [0.77, 0.25, 0.10],  # one      (x^1)
    [0.05, 0.80, 0.55]]  # step     (x^1)
)

query = inputs[1] # this token is taken as our query token example
# attention scores for the qery token are calculated as the dot product between the token itself and all other token in the sequence
# the higher the score, the greatest the importance
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

# attention scores are normalized to obtain attention weights (that sum up to 1)
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# basic implementation of softmax function, commonly used to normalize the attention scores
# it ensures that weights are always positive
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)

## 3.3.2 Computing attention weights for all input tokens
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

# we reach the same result using more efficient matrix multiplication
attn_scores = inputs @ inputs.T
print(attn_scores)

# dim=-1 means apply normalization along last dimension of the tensor
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# check that the second row corresponds to previously calculated context
print("Previous 2nd cotext vector:", context_vec_2)

## 3.4 Implementing self-attention with trainable weights
x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # input embedding size (d=3)
d_out = 2 # output embedding size (d_out=2)

# initialize the three weighting matrices
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# compute query, key and value vectors
query_2 = x_2 @ W_query
key_2   = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# key and value vectors for all input elements are required for computing the attention weights with respect to the query q_2
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# we can calculate the attention score with respect to query 2 is calculcated by dot product of q_2 with the keys
keys_2 = keys[1]
attn_scores_22 = query_2.dot(keys_2)
print(attn_scores_22) # (unnormalized) attention score only for element 2

# all attention scores given query 2
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

# from (unscaled) attention scores to (scaled) attention weights using softmax
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) # the scale by embedding dimension of the keys is to prevent small gradients
print(attn_weights_2)

# final step: calculathe the context vectors
context_vec_2 = attn_weights_2 @ values # the attention weights serve as a weighting factor that weighs the repsective importance of each value vector
print(context_vec_2)

# compact self-attention class
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    # forward pass function
    def forward(self, x):
        keys    = x @ self.W_key
        queries = x @ self.W_query
        values  = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# improved self-attention class using PyTorch's Linear layers (which provides a more sophisticated weight initialization scheme)
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    # forward pass function
    def forward(self, x):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

## 3.5 Hiding future words with causal attention