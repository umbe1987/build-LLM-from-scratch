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
