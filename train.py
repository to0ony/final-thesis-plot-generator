import os
import pickle
import time
import datetime
import math
import torch
import numpy as np
import csv
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from mingpt.model import GPT
import tiktoken
from config import *
from torch.utils.tensorboard import SummaryWriter


# -------------------------
# TensorBoard
# -------------------------

run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
tb_dir = os.path.join('runs', run_name)
tb_writer = SummaryWriter(tb_dir)
print(f"TensorBoard logging: {tb_dir}")

# -------------------------
# Hiperparametri 
# -------------------------
batch_size = BATCH_SIZE
gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
block_size = BLOCK_SIZE
max_iters = MAX_ITERS
eval_interval = EVAL_INTERVAL
learning_rate = LEARNING_RATE
eval_iters = EVAL_ITERS
max_grad_norm = MAX_GRAD_NORM

# CUDA provjera
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("GPU: " + torch.cuda.get_device_name(0)) if device == 'cuda' else print("GPU nije dostupan - koristi se CPU")

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
train_data = np.memmap('dataset/processed/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('dataset/processed/val.bin', dtype=np.uint16, mode='r')

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
            with autocast(device_type=device):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def generate_example(prompt="A mysterious murder shocks the town when ", max_new_tokens=150):
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    generated = model.generate(context, max_new_tokens=max_new_tokens)
    return decode(generated[0].tolist())

# -------------------------
# Model
# -------------------------
config = get_model_config(vocab_size)
model = GPT(config).to(device)

# -------------------------
# Model info
# -------------------------
total_params = sum(p.numel() for p in model.parameters())
print(f"Pokretanje treniranja na: {device.upper()}")
print(f"Model parametri: {total_params:,}")
print(f"Ukupno iteracija: {max_iters:,}")

tb_writer.add_text("run/meta", f"device: {device}, params: {total_params:,}, block_size: {block_size}, "
                               f"batch_size: {batch_size}, grad_accum: {gradient_accumulation_steps}, "
                               f"lr: {learning_rate}, warmup: {WARMUP_STEPS}")

# Log additional metadata
metadata = f"eval_interval: {eval_interval}, eval_iters: {eval_iters}, max_grad_norm: {max_grad_norm}"
tb_writer.add_text("run/extended_meta", metadata)

# Log model architecture
tb_writer.add_text("run/model_architecture", str(model))

# -------------------------
# Optimizator i warmup scheduler
# -------------------------
# AdamW s weight decay i warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

# Warmup scheduler za stabilno treniranje
warmup_steps = WARMUP_STEPS
def get_lr(iter):
    if iter < warmup_steps:
        return learning_rate * iter / warmup_steps
    else:
        # Cosine annealing nakon warmupa
        return learning_rate * 0.5 * (1 + math.cos(math.pi * (iter - warmup_steps) / (max_iters - warmup_steps)))

scaler = GradScaler(device)

# -------------------------
# Trening 
# -------------------------

# Early stopping parameters
patience = 5  # Number of evaluations to wait for improvement
best_val_loss = float('inf')
no_improvement_count = 0

# Retention policy: Save only the best-performing checkpoint
best_checkpoint_path = None

# Retention policy: Keep only the last three checkpoints
checkpoint_paths = []

os.makedirs('models/checkpoints', exist_ok=True)

try:
    for iter in range(max_iters):
        t0 = time.time()

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # TensorBoard: losses
            tb_writer.add_scalar("loss/train", float(losses['train']), iter)
            tb_writer.add_scalar("loss/val", float(losses['val']), iter)

            tb_writer.add_scalars("loss/overview", {
            "train": float(losses['train']),
            "val":   float(losses['val']),
            }, iter)

            # TensorBoard: short generated sample
            sample_text = generate_example()
            tb_writer.add_text("samples/generation", sample_text, iter)

            # Checkpointing
            checkpoint_path = None  # Initialize checkpoint_path to avoid NameError

            if iter != 0:
                checkpoint_path = f"models/checkpoint-{iter:05d}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'step': iter
                }, checkpoint_path)
                print(f"Checkpoint spremljen: {checkpoint_path}")

                # Checkpoint retention policy
                checkpoint_paths.append(checkpoint_path)
                if len(checkpoint_paths) > 3:
                    oldest_checkpoint = checkpoint_paths.pop(0)
                    if os.path.exists(oldest_checkpoint):
                        os.remove(oldest_checkpoint)
                        print(f"Old checkpoint removed: {oldest_checkpoint}")

            # Early stopping check
            val_loss = losses['val']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0  # Reset counter if there is an improvement

                best_checkpoint_path = checkpoint_path

            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print("Early stopping triggered. No improvement in validation loss.")
                break

        # Gradient accumulation
        loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(gradient_accumulation_steps):
            xb, yb = get_batch('train')
            with autocast(device_type=device):
                logits, loss = model(xb, yb)
                loss = loss / gradient_accumulation_steps
            loss_accum += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        current_lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(iter)
        tb_writer.add_scalar("lr", current_lr, iter)

        # Log gradients of model parameters
        for name, param in model.named_parameters():
            if param.grad is not None:
                tb_writer.add_histogram(f"gradients/{name}", param.grad, iter)

        # Log model weights
        for name, param in model.named_parameters():
            tb_writer.add_histogram(f"weights/{name}", param, iter)

        dt = time.time() - t0
        # Log training time
        tb_writer.add_scalar("time/iteration", dt, iter)
        if iter % 100 == 0:
            print(f"step {iter} | loss: {loss_accum:.4f} | lr: {get_lr(iter):.6f} | time: {dt:.2f}s")

except KeyboardInterrupt:
    last_ckpt = "models/checkpoint-last.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'step': iter
    }, last_ckpt)
    print(f"\nTrening prekinut. Model spremljen kao {last_ckpt}")
    tb_writer.close()
else:
    tb_writer.close()

if best_checkpoint_path is not None:
    print(f"Najbolji checkpoint spremljen kao: {best_checkpoint_path}")
