import os
import pickle
import time
import math
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from mingpt.model import GPT
import tiktoken

# -------------------------
# Hiperparametri
# -------------------------
batch_size = 16
gradient_accumulation_steps = 4
block_size = 256
max_iters = 8000
eval_interval = 400
learning_rate = 3e-4
eval_iters = 50
n_embd = 768
n_layer = 12      
n_head = 12
max_grad_norm = 1.0

# CUDA provjera
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Koristi GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("GPU nije dostupan - koristi se CPU")

# -------------------------
# Vokabular
# -------------------------
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

def encode(s):
    return enc.encode(s)

def decode(tokens):
    return enc.decode(tokens)

# -------------------------
# Podaci
# -------------------------
train_data = np.memmap('dataset/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('dataset/val.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with autocast(device):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def generate_example(prompt="A mysterious murder shocks the town when ", max_new_tokens=100):
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    generated = model.generate(context, max_new_tokens=max_new_tokens)
    print("\n[GENERATED SAMPLE]\n" + decode(generated[0].tolist()) + "\n")

# -------------------------
# Model
# -------------------------
config = GPT.get_default_config()
config.vocab_size = vocab_size
config.block_size = block_size
config.n_layer = n_layer
config.n_head = n_head
config.n_embd = n_embd
config.model_type = None

model = GPT(config).to(device)

# -------------------------
# Model info
# -------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Pokretanje treniranja na: {device.upper()}")
print(f"Model parametri: {total_params:,} (trenabilni: {trainable_params:,})")
print(f"Ukupno iteracija: {max_iters:,}")

# -------------------------
# Optimizator i scheduler - GPT-2 style
# -------------------------
# AdamW s weight decay i warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

# Warmup scheduler za stabilno treniranje
warmup_steps = 500
def get_lr(iter):
    if iter < warmup_steps:
        return learning_rate * iter / warmup_steps
    else:
        # Cosine annealing nakon warmupa
        return learning_rate * 0.5 * (1 + math.cos(math.pi * (iter - warmup_steps) / (max_iters - warmup_steps)))

scaler = GradScaler(device)

# -------------------------
# Trening s gradient accumulation
# -------------------------
for iter in range(max_iters):
    t0 = time.time()

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        generate_example()

    # Gradient accumulation za veći efektivni batch
    loss_accum = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        
        with autocast(device):
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps  # Scale loss
            
        loss_accum += loss.item()
        scaler.scale(loss).backward()

    # Optimizator korak nakon svih micro-batch-eva
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    
    # Ažuriraj learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(iter)

    dt = time.time() - t0
    if iter % 100 == 0:
        print(f"step {iter} | loss: {loss_accum:.4f} | lr: {get_lr(iter):.6f} | time: {dt:.2f}s")

# -------------------------
# Spremi checkpoint
# -------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'step': max_iters
}, 'checkpoint.pt')

print("Model spremljen kao checkpoint.pt")
