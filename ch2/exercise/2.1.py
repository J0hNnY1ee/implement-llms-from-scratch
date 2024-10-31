import os, tiktoken

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

tokenizer = tiktoken.get_encoding("gpt2")
unk = 'Akwirw ier'
ids= tokenizer.encode(unk)
print(ids )
print(tokenizer.decode(ids))
