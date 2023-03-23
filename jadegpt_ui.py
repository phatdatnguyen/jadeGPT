import os
import json
import jadegpt
import gradio as gr

config = { 'data_dir':'C:\\data',
          'model_dir':'C:\\model',
          'device':'GPU',
          'dtype':'bfloat16'
        }
config_file_path = os.path.abspath('') + '\\config.json'
model = None
model_finetune = None

def load_settings():
    global config
    with open(config_file_path, 'r', encoding='utf8') as f:
        config = json.load(f)
    print('Settings loaded')

def save_settings(data_dir, model_dir, device, dtype):
    global config
    with open(config_file_path, 'w', encoding='utf8') as f:
        config['data_dir'] = data_dir
        config['model_dir'] = model_dir
        config['device'] = device
        config['dtype'] = dtype
        json.dump(config, f)
    return 'Settings saved'

if os.path.exists(config_file_path):
    load_settings()

def load_data(input_file, split, use_gpt2_encoding, data_dir):
    input_file_path = input_file.name
    data = jadegpt.open_dataset_file(input_file_path)
    train_data, val_data = jadegpt.split_dataset(data, split)
    jadegpt.export_data_to_files(data, train_data, val_data, use_gpt2_encoding, data_dir, 'train.bin', 'val.bin', 'meta.pkl')
    output = 'train.bin and val.bin were saved to ' + data_dir
    if use_gpt2_encoding==False:
        output += '\nmeta.pkl was saved to ' + data_dir
    vocab_size = jadegpt.get_vocab_size(data, use_gpt2_encoding)
    return output, vocab_size

def init_gpt_model_for_training(random_seed, n_layer, n_head, n_embd, dropout, bias, block_size, vocab_size):
    global model
    model = jadegpt.init_gpt(random_seed, n_layer, n_head, n_embd, dropout, bias, block_size, int(vocab_size))
    return 'GPT model was initialized!'

def init_ckpt_for_finetuning(ckpt_file_finetune, random_seed):
    global model_finetune
    ckpt_file_path = ckpt_file_finetune.name
    device_used = 'cuda' if device == 'GPU' else 'cpu'
    model_finetune = jadegpt.resume_gpt(ckpt_file_path, random_seed, device_used)
    return 'GPT model was loaded from a checkpoint!'

def init_gpt2_model_for_finetuning(gpt2_model, random_seed):
    global model_finetune
    model_finetune = jadegpt.init_gpt2(gpt2_model.lower(), random_seed)
    return 'GPT2 model was loaded!'

def train_gpt(dtype, device, block_size, batch_size,\
                  max_iters, learning_rate,\
                  decay_lr, \
                  gradient_accumulation_steps, log_interval,\
                  only_save_on_finish, save_interval, model_dir, model_name):
    train_data = jadegpt.load_data_file_to_memmap(config['data_dir'], 'train.bin')
    val_data = jadegpt.load_data_file_to_memmap(config['data_dir'], 'val.bin')
    device_used = 'cuda' if device == 'GPU' else 'cpu'
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.99
    warmup_iters = 0
    lr_decay_iters = max_iters
    min_lr = learning_rate / 10.0
    eval_interval = 50
    eval_iters = 20
    grad_clip = 1.0
    jadegpt.train_gpt(model, dtype, device_used, train_data, val_data, block_size, batch_size,\
                  max_iters, weight_decay, learning_rate, beta1, beta2, warmup_iters,\
                  lr_decay_iters, min_lr, decay_lr, eval_interval, eval_iters,\
                  gradient_accumulation_steps, grad_clip, log_interval,\
                  only_save_on_finish, save_interval, model_dir, model_name)
    return 'GPT model was trained and saved to ' + model_dir

def finetune_gpt(dtype, device, batch_size,\
                  max_iters, learning_rate,\
                  decay_lr, \
                  gradient_accumulation_steps, log_interval,\
                  only_save_on_finish, save_interval, model_dir, model_name):
    train_data = jadegpt.load_data_file_to_memmap(config['data_dir'], 'train.bin')
    val_data = jadegpt.load_data_file_to_memmap(config['data_dir'], 'val.bin')
    block_size = model_finetune.config.block_size
    device_used = 'cuda' if device == 'GPU' else 'cpu'
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.99
    warmup_iters = 0
    lr_decay_iters = max_iters
    min_lr = learning_rate / 10.0
    eval_interval = 50
    eval_iters = 20
    grad_clip = 1.0
    jadegpt.train_gpt(model_finetune, dtype, device_used, train_data, val_data, block_size, batch_size,\
                  max_iters, weight_decay, learning_rate, beta1, beta2, warmup_iters,\
                  lr_decay_iters, min_lr, decay_lr, eval_interval, eval_iters,\
                  gradient_accumulation_steps, grad_clip, log_interval,\
                  only_save_on_finish, save_interval, model_dir, model_name)
    return 'GPT model was fine-tuned and saved to ' + model_dir

def generate_from_trained_gpt(prompt, use_gpt2_encoding, num_samples, max_new_tokens, temperature, top_k, device, dtype):
    meta_dir = config['data_dir']
    meta_file_name = 'meta.pkl'
    device_used = 'cuda' if device == 'GPU' else 'cpu'
    output = jadegpt.generate_text(model, prompt, use_gpt2_encoding, meta_dir, meta_file_name, num_samples, max_new_tokens, temperature, top_k, device_used, dtype)
    return output

def generate_from_finetuned_gpt(prompt, use_gpt2_encoding, num_samples, max_new_tokens, temperature, top_k, device, dtype):
    meta_dir = config['data_dir']
    meta_file_name = 'meta.pkl'
    device_used = 'cuda' if device == 'GPU' else 'cpu'
    output = jadegpt.generate_text(model_finetune, prompt, use_gpt2_encoding, meta_dir, meta_file_name, num_samples, max_new_tokens, temperature, top_k, device_used, dtype)
    return output

with gr.Blocks(title='jadeGPT') as ui:
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown('# jadeGPT')
        with gr.Column(scale=2, min_width=600):
            with gr.Accordion(label='Settings', open=False):
                with gr.Row():
                    with gr.Column():
                        data_dir = gr.Textbox(value=config['data_dir'], label='Data folder')
                        model_dir = gr.Textbox(value=config['model_dir'], label='Model folder')
                    with gr.Column():
                        device = gr.Dropdown(choices=['GPU', 'CPU'], value=config['device'], label='Device')
                        dtype = gr.Dropdown(choices=['bfloat16', 'float16', 'float32'], value=config['dtype'], label='Data type')
                save_settings_button = gr.Button(value='Save settings')
                save_settings_result = gr.Markdown()
                save_settings_button.click(save_settings, [data_dir, model_dir, device, dtype], save_settings_result)
    with gr.Tab('Train'):
        with gr.Row():
            with gr.Column():
                gr.Markdown('## Training data')
                input_file = gr.File(file_types=["text"], label='Input text file')
                split_ratio = gr.Slider(minimum=0.01, maximum=0.99, value=0.9, step=0.01, label='Split ratio')
                use_gpt2_encoding = gr.Checkbox(value=False, label='Use GPT2 encoding')
                load_data_button = gr.Button(value='Encode and split dataset')
                load_data_result = gr.Markdown()
                vocab_size = gr.Textbox(label='Vocabulary size')
                load_data_button.click(load_data, [input_file, split_ratio, use_gpt2_encoding, data_dir], [load_data_result, vocab_size])
            with gr.Column():
                gr.Markdown('## Initialize GPT model')
                n_layer = gr.Slider(minimum=1, maximum=24, value=6, step=1.0, label='Number of layers')
                n_head = gr.Slider(minimum=1, maximum=24, value=6, step=1.0, label='Number of attention heads')
                n_embd = gr.Slider(minimum=1, maximum=768, value=384, step=2.0, label='Number of embeddings')
                block_size = gr.Slider(minimum=1, maximum=1024, value=32, step=16.0, label='Block size')
                dropout = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, label='Dropout ratio')
                bias = gr.Checkbox(value=False, label='Use bias')
                random_seed = gr.Textbox(value=1337, label='Random seed')
                init_gpt_button = gr.Button(value='Initialize GPT model')
                init_model_result = gr.Markdown()
                init_gpt_button.click(init_gpt_model_for_training, [random_seed, n_layer, n_head, n_embd, dropout, bias, block_size, vocab_size], init_model_result)
            with gr.Column():
                gr.Markdown('## Training')
                model_name = gr.Textbox(value='model', label='Model name')
                batch_size = gr.Slider(minimum=1, maximum=24, value=8, step=1.0, label='Batch size')
                gradient_accumulation_steps = gr.Slider(minimum=1, maximum=32, value=5, step=1.0, label='Gradient accumulation steps')
                learning_rate = gr.Slider(minimum=1e-4, maximum=15-3, value=1e-3, step=1e-4, label='Learning rate')
                max_iters = gr.Slider(minimum=1, maximum=100000, value=100, step=1.0, label='Number of iterations')
                decay_lr = gr.Checkbox(value=True, label='Decay learning rate')
                log_interval = gr.Slider(minimum=1, maximum=100, value=10, step=16.0, label='Log interval')
                only_save_on_finish = gr.Checkbox(value=False, label='Only save checkpoint when finish training')
                save_interval = gr.Slider(minimum=1, maximum=100, value=50, step=16.0, label='Save checkpoint interval')
                train_button = gr.Button(value='Train GPT model')
                train_result = gr.Markdown()
                train_button.click(train_gpt, [dtype, device, block_size, batch_size, max_iters, learning_rate, decay_lr, gradient_accumulation_steps, log_interval, only_save_on_finish, save_interval, model_dir, model_name], train_result)
        with gr.Row():
            gr.Markdown('## Text generation')
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label='Prompt', lines=5)
                num_samples = gr.Slider(minimum=1, maximum=5, value=3, step=1.0, label='Number of samples to generate')
                max_new_tokens = gr.Slider(minimum=1, maximum=500, value=100, step=1.0, label='Number of characters to generate')    
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.01, label='Temperature')  
                top_k = gr.Slider(minimum=0, maximum=100, value=20, step=1.0, label='Top k')
                generate_button = gr.Button(value='Generate')
            with gr.Column():
                output = gr.Textbox(label='Output', lines=15)    
                generate_button.click(generate_from_trained_gpt, [prompt, use_gpt2_encoding, num_samples, max_new_tokens, temperature, top_k, device, dtype], output)
    with gr.Tab('Fine-tune'):
        with gr.Row():
            with gr.Column():
                gr.Markdown('## Fine-tuning data')
                input_file_finetune = gr.File(file_types=["text"], label='Input text file')
                split_ratio_finetune = gr.Slider(minimum=0.01, maximum=0.99, value=0.9, step=0.01, label='Split ratio')
                use_gpt2_encoding_finetune = gr.Checkbox(value=False, label='Use GPT2 encoding')
                load_data_button_finetune = gr.Button(value='Encode and split dataset')
                load_data_result_finetune = gr.Markdown()
                vocab_size_finetune = gr.Textbox(label='Vocabulary size')
                load_data_button_finetune.click(load_data, [input_file_finetune, split_ratio_finetune, use_gpt2_encoding_finetune, data_dir], [load_data_result_finetune, vocab_size_finetune])
            with gr.Column():
                gr.Markdown('## GPT model')
                with gr.Tab('Load from a checkpoint'):
                    ckpt_file_finetune = gr.File(file_types=[".ckpt"], label='Checkpoint file')
                    random_seed_ckpt_finetune = gr.Textbox(value=1337, label='Random seed')
                    init_ckpt_button = gr.Button(value='Load checkpoint')
                    init_ckpt_result = gr.Markdown()
                    init_ckpt_button.click(init_ckpt_for_finetuning, [ckpt_file_finetune, random_seed], init_ckpt_result)
                with gr.Tab('Pretrained GPT2 model'):
                    gpt2_model = gr.Dropdown(choices=['GPT2', 'GPT2-medium', 'GPT2-large', 'GPT2-xl'], value='GPT2', label='GPT2 model')
                    random_seed_gpt2_finetune = gr.Textbox(value=1337, label='Random seed')  
                    init_gpt2_button = gr.Button(value='Load GPT2 model')
                    init_gpt2_result = gr.Markdown()
                    init_gpt2_button.click(init_gpt2_model_for_finetuning, [gpt2_model, random_seed], init_gpt2_result)
            with gr.Column():
                gr.Markdown('## Fine-tuning')
                model_name_finetune = gr.Textbox(value='model', label='Model name')
                batch_size_finetune = gr.Slider(minimum=1, maximum=24, value=8, step=1.0, label='Batch size')
                gradient_accumulation_steps_finetune = gr.Slider(minimum=1, maximum=32, value=5, step=1.0, label='Gradient accumulation steps')
                learning_rate_finetune = gr.Slider(minimum=1e-4, maximum=15-3, value=1e-3, step=1e-4, label='Learning rate')
                max_iters_finetune = gr.Slider(minimum=1, maximum=100000, value=100, step=1.0, label='Number of iterations')
                decay_lr_finetune = gr.Checkbox(value=True, label='Decay learning rate')
                log_interval_finetune = gr.Slider(minimum=1, maximum=100, value=10, step=16.0, label='Log interval')
                only_save_on_finish_finetune = gr.Checkbox(value=False, label='Only save checkpoint when finish training')
                save_interval_finetune = gr.Slider(minimum=1, maximum=100, value=50, step=16.0, label='Save checkpoint interval')
                finetune_button = gr.Button(value='Train GPT model')
                finetune_result = gr.Markdown()
                finetune_button.click(finetune_gpt, [dtype, device, batch_size_finetune, max_iters_finetune, learning_rate_finetune, decay_lr_finetune, gradient_accumulation_steps_finetune, log_interval_finetune, only_save_on_finish_finetune, save_interval_finetune, model_dir, model_name_finetune], finetune_result)
        with gr.Row():
            gr.Markdown('## Text generation')
        with gr.Row():
            with gr.Column():
                prompt_finetune = gr.Textbox(label='Prompt', lines=5)
                num_samples_finetune = gr.Slider(minimum=1, maximum=5, value=3, step=1.0, label='Number of samples to generate')
                max_new_tokens_finetune = gr.Slider(minimum=1, maximum=500, value=100, step=1.0, label='Number of characters to generate')    
                temperature_finetune = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.01, label='Temperature')  
                top_k_finetune = gr.Slider(minimum=0, maximum=100, value=20, step=1.0, label='Top k')
                generate_button_finetune = gr.Button(value='Generate')
            with gr.Column():
                output_finetune = gr.Textbox(label='Output', lines=15)    
                generate_button_finetune.click(generate_from_finetuned_gpt, [prompt_finetune, use_gpt2_encoding_finetune, num_samples_finetune, max_new_tokens_finetune, temperature_finetune, top_k_finetune, device, dtype], output_finetune)

ui.launch()