## Import libraries
import os
import requests
import tiktoken
import numpy as np
import time
import math
import pickle
from contextlib import nullcontext
import torch

from model import GPTConfig, GPT

def open_dataset_file(input_dir, data_file_name):
    input_file_path = input_dir + '\\' + data_file_name
    with open(input_file_path, 'r', encoding='utf8') as f:
        data = f.read()
    return data

def split_dataset(data, split):
    n = len(data)
    train_data = data[:int(n*split)]
    val_data = data[int(n*split):] 
    return train_data, val_data

def get_vocab_size(data):
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")
    return vocab_size

def export_data_to_files(data, train_data, val_data, use_gpt2_encoding, data_dir, train_file_name, val_file_name, meta_file_name):
    # create a mapping from characters to integers
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    if use_gpt2_encoding:
        encoding = 'gpt2'
        stoi = {}
        itos = {}
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

    else:
        encoding = 'custom'
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        def encode(s):
            return [stoi[c] for c in s] # encoder: take a string, output a list of integers

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train dataset has {len(train_ids):,} tokens")
    print(f"val dataset has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(data_dir + '\\' + train_file_name)
    print(f"{train_file_name} was saved to {data_dir}")
    val_ids.tofile(data_dir + '\\' + val_file_name)
    print(f"{val_file_name} was saved to {data_dir}")

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'encoding': encoding,
        'itos': itos,
        'stoi': stoi,
    }
    with open(data_dir + '\\' + meta_file_name, 'wb') as f:
        pickle.dump(meta, f)
    print(f"{meta_file_name} was saved to {data_dir}")

def load_data_file_to_memmap(data_dir, data_file_name):
    data = np.memmap(data_dir + '\\' + data_file_name, dtype=np.uint16, mode='r')
    return data

def init_gpt(random_seed = 1337, n_layer = 6, n_head = 6, n_embd = 384, dropout = 0.0, bias = False, block_size = 32, vocab_size = 50304):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=vocab_size, dropout=dropout) # start with model_args from command line
    # init a new model from scratch
    print("Initializing a new GPT model from scratch")
    # determine the vocab size we'll use for from-scratch training
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)

    return model

def init_gpt2(gpt2_model = 'gpt2', random_seed = 1337):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    
    # init from a given GPT-2 model
    print("Initializing a GPT-2 model")
    model = GPT.from_pretrained(gpt2_model, dict(dropout=0.0))
    model.eval()

    return model

def resume_gpt(model_dir, model_file_name, random_seed, device):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    
    checkpoint = torch.load(model_dir + '\\' + model_file_name, map_location=device)

    # resume from a checkpoint
    print("Initializing a GPT model from a checkpoint")
    gptconf = checkpoint['model_args']
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()

    return model

def get_batch(split, device, block_size, batch_size):
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ix = torch.randint(len(split) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((split[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((split[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, eval_iters, ctx, train_data, val_data, device, block_size, batch_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == 'train':
                X, Y = get_batch(train_data, device, block_size, batch_size)
            else: # split == 'val'
                X, Y = get_batch(val_data, device, block_size, batch_size)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train_gpt(model, dtype, device, train_data, val_data, block_size, batch_size, max_iters, weight_decay, learning_rate, beta1, beta2, warmup_iters, lr_decay_iters, min_lr, decay_lr, eval_interval, eval_iters, gradient_accumulation_steps, grad_clip, log_interval, only_save_on_finish, save_interval, model_dir, model_name):
    # init these up here
    iter_num = 1
    best_val_loss = 1e9
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    gradient_accumulation_steps *= 8

    # send model to device    
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    # training loop
    X, Y = get_batch(train_data, device, block_size, batch_size) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 1 # number of iterations in the lifetime of this process
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model, eval_iters, ctx, train_data, val_data, device, block_size, batch_size)

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(train_data, device, block_size, batch_size)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        # saving checkpoints
        if iter_num % save_interval == 0 and only_save_on_finish == False and iter_num != max_iters:
            save_checkpoint(model, optimizer, iter_num, best_val_loss, model_dir, model_name + '-' + str(iter_num) + '.cpkt')

        # termination conditions
        if iter_num == max_iters:
            save_checkpoint(model, optimizer, iter_num, best_val_loss, model_dir, model_name + '-' + str(max_iters) + '.cpkt')
            break
        
        iter_num += 1
        local_iter_num += 1

def save_checkpoint(model, optimizer, iter_num, best_val_loss, model_dir, model_name='ckpt.pt'):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model.config,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss
    }
    os.makedirs(model_dir, exist_ok=True)
    torch.save(checkpoint, model_dir + '\\' + model_name)
    print(f"gpt model was saved to {model_dir}\\{model_name}")
    
def generate_text(model, start, use_gpt2_encoding, meta_dir, meta_file_name, num_samples, max_new_tokens, temperature, top_k, device, dtype):
    if use_gpt2_encoding:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    else:
        print(f"Loading meta from {meta_dir}\\{meta_file_name}...")
        with open(meta_dir + '\\' + meta_file_name, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    
    model.to(device)
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')