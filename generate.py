import torch
import tiktoken
from mingpt.model import GPT

# -------------------------
# Setup
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------
# Tokenizer (tiktoken GPT-2)
# -------------------------
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

def encode(s):
    return torch.tensor([enc.encode(s)], dtype=torch.long).to(device)

def decode(l):
    return enc.decode(l)

# -------------------------
# Konfiguracija modela (mora biti ista kao u train.py)
# -------------------------
config = GPT.get_default_config()
config.vocab_size = vocab_size
config.block_size = 256        
config.n_layer = 12             
config.n_head = 12              
config.n_embd = 768             
config.model_type = None

# -------------------------
# Inicijalizacija modela
# -------------------------
model = GPT(config)
checkpoint = torch.load('checkpoint.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# -------------------------
# Prompt i generacija
# -------------------------
prompts = [
    "Film"
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n PLOT #{i}:")
    print(f" PROMPT: {prompt}")
    print("-" * 60)
    
    x = encode(prompt)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=200, temperature=0.7, do_sample=True, top_k=50)
    
    generated = decode(y[0].tolist())
    print(f" STORY: {generated}")
    print("=" * 80)
