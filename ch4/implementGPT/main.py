import torch, tiktoken, os
from FeedForward import FeedForward
from MultiHeadAttention import MultiHeadAttention
from TransformerBlock import TransformerBlock
from config import GPT_CONFIG_124M
from GPTModel import GPTModel

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"






def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 加了一个维度
    print("encoded_tensor:", encoded_tensor)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    out = generate_text_simple(model,encoded_tensor,6,GPT_CONFIG_124M["context_length"])
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
    