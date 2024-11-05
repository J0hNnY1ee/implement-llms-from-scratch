import torch
from FeedForward import FeedForward
from MultiHeadAttention import MultiHeadAttention
from TransformerBlock import TransformerBlock
from config import GPT_CONFIG_124M

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
