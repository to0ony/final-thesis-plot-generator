import torch
import tiktoken
from mingpt.model import GPT
from config import get_model_config

# -------------------------
# Setup
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (torch.cuda.get_device_name(0)) if device == 'cuda' else print("GPU nije dostupan - koristi se CPU")

# -------------------------
# Tokenizer
# -------------------------
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

def encode(s):
    return torch.tensor([enc.encode(s)], dtype=torch.long).to(device)

def decode(l):
    return enc.decode(l)

# -------------------------
# Konfiguracija modela
# -------------------------
config = get_model_config(vocab_size)

# -------------------------
# Inicijalizacija modela
# -------------------------
model = GPT(config)
checkpoint = torch.load('models/cmu_plots_checkpoint.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# -------------------------
# Prompt i generacija
# -------------------------
prompts = [
    "A little young boy manages to enter TV as portal"
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

torch.cuda.empty_cache()