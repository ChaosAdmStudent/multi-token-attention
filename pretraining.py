import torch
import torch.nn as nn 
import torch.nn.functional as F 
from tqdm import tqdm
from model import GPTModel
from utils import load_weights_into_gpt, split_txt_train_test_val, plot_values, generate_out_text
from gpt_download import download_and_load_gpt2
from mta import MultiTokenAttention
from data_prep import create_dataloader, prepare_babylm_dataset
import tiktoken 
import os 

def calc_loss_loader(model, data_loader, device, eval_batches=None): 
    model.eval() 
    model = model.to(device) 
    total_loss = 0.0
    eval_batches = len(data_loader) if eval_batches is None else eval_batches 

    for i, (x_train, y_train) in tqdm(enumerate(data_loader)): 
        x_train = x_train.to(device) 
        y_train = y_train.to(device) 

        if i < eval_batches: 
            with torch.no_grad(): 
                logits: torch.Tensor = model(x_train)

            loss = F.cross_entropy(logits, y_train) 
            total_loss += loss.item() 
    
    return total_loss / eval_batches

def evaluate_model(train_loader, val_loader, model, device, eval_batches=None): 
    model.eval() 
    train_loss = calc_loss_loader(model, train_loader, device, eval_batches) 
    val_loss = calc_loss_loader(model, val_loader, device, eval_batches) 
    model.train() 
    return train_loss, val_loss 


def train_model(model, optimizer, num_epochs, train_loader, val_loader, device, 
                start_tokens, tokenizer, eval_freq=None, eval_batches=None): 
    
    model = model.to(device) 
    global_step, total_examples_seen = -1, 0
    train_losses, val_losses = [] , [] 
    eval_freq = 50 if eval_freq is None else eval_freq

    for epoch in tqdm(range(num_epochs)):     
        for i, (x_train, y_train) in tqdm(enumerate(train_loader), leave=False): 
            model.train() 
            x_train: torch.Tensor = x_train.to(device) 
            y_train: torch.Tensor = y_train.to(device) 
            logits = model(x_train) # (B,N,vocab_size) 
            loss = F.cross_entropy(logits.flatten(0,1), y_train.flatten()) # Logits shape: (B*N, vocab_size); target shape: (B*N,)  

            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad() 

            global_step += 1 
            total_examples_seen += x_train.shape[0] 

            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(train_loader, val_loader, model, device, eval_batches) 
                train_losses.append(train_loss) 
                val_losses.append(val_loss) 
                print(f'\t Epoch {epoch} step {i}/{len(train_loader)} train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}')

            if global_step % 10000 == 0: 
                # Save model 
                torch.save(model.state_dict(), 'model_mta.pth')
                
        # Check how start_context is being replied to after each epoch 
        output_text = generate_out_text(35, model, start_tokens, tokenizer, model.context_length, device)
        print(f'#########Epoch {epoch}##############') 
        print(f'{output_text}')
        print(f'#########Epoch {epoch}##############')
    
    return train_losses, val_losses,  total_examples_seen

def main(): 

    # Prepare BabyLM dataset
    data_folder = "data"
    babylm_file = os.path.join(data_folder, "babylm.txt")
    prepare_babylm_dataset(data_folder, babylm_file)
    
    model_configs = {
        "gpt2-small (124M)": {"token_emb_dim": 768, "kv_dim": 768, "num_layers": 12, "num_heads": 12},
        "gpt2-medium (355M)": {"token_emb_dim": 1024, "kv_dim": 768, "num_layers": 24, "num_heads": 16},
        "gpt2-large (774M)": {"token_emb_dim": 1280, "kv_dim": 768, "num_layers": 36, "num_heads": 20},
        "gpt2-xl (1558M)": {"token_emb_dim": 1600, "kv_dim": 768, "num_layers": 48, "num_heads": 25},
    }

    conv_configs = {
        "kernel_size": (5,5,4), # (c_q, c_k, c_h)
        "conv_option": "pre", # 'pre', 'post', 'mix' 
    }

    BASE_CONFIG = {
        "vocab_size": 50257, 
        "context_length": 1024, 
        "droprate": 0.2, 
        "qkv_bias": True, 
        "max_batch_size": 8, 
        "max_seq_len": 1024,
    } 

    BASE_CONFIG.update(model_configs["gpt2-small (124M)"])  
    BASE_CONFIG.update(conv_configs)

    print('Initializing MTA Model')
    model_mta = GPTModel(BASE_CONFIG, "mta") 
    # model_mha = GPTModel(BASE_CONFIG, "mha") 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    print(f'Using device: {device}')

    # Download and load weights
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="pretraining/gpt2") 
    load_weights_into_gpt(model_mta, params) 
    # load_weights_into_gpt(model_mha, params) 

    # 1. Freeze all weights for MTA model
    for param in model_mta.parameters(): 
        param.requires_grad = False 
    
    # 2. Unfreeze only convolution parameters in MTA
    for block in model_mta.trf_blocks:
        if isinstance(block.att, MultiTokenAttention):
            if block.att.conv_option in ['pre', 'post']:
                for param in block.att.conv.parameters():
                    param.requires_grad = True
            elif block.att.conv_option == 'mix':
                for param in block.att.conv_qk.parameters():
                    param.requires_grad = True
                for param in block.att.conv_heads.parameters():
                    param.requires_grad = True 
    
    print('Unfroze only convolution parameters in MTA model.')

    # Load data loaders 
    data_folder = "data"
    sample_file = os.path.join(data_folder, "simple_wiki.train")
    with open(sample_file, 'r') as file: 
        txt = file.read() 
    
    train_txt, val_txt, test_txt = split_txt_train_test_val(txt, train_size=0.8, val_size=0.1)
    tokenizer = tiktoken.get_encoding('gpt2')
    train_loader = create_dataloader(
        train_txt, 
        tokenizer,
        max_length=BASE_CONFIG["context_length"],
        stride=BASE_CONFIG["context_length"],
        batch_size=4,
        shuffle=True,
        drop_last=True,
        num_workers=0)
    
    val_loader = create_dataloader(
        val_txt, 
        tokenizer,
        max_length=BASE_CONFIG["context_length"],
        stride=BASE_CONFIG["context_length"],
        batch_size=4,
        shuffle=False,
        drop_last=True,
        num_workers=0)
    

    test_loader = create_dataloader(
        test_txt, 
        tokenizer,
        max_length=BASE_CONFIG["context_length"],
        stride=BASE_CONFIG["context_length"],
        batch_size=4,
        shuffle=False,
        drop_last=True,
        num_workers=0)
    
    print('Train loader size:', len(train_loader))
    print('Validation loader size:', len(val_loader))
    print('Test loader size:', len(test_loader))

    # Set up training parameters
    optimizer = torch.optim.AdamW(
        model_mta.parameters(), 
        lr=1e-4, 
        weight_decay=0.01
    )
    num_epochs = 3
    start_context = 'The biggest problem with' 

    print('###################################')
    print('Started training: ') 
    print('Monitoring given sentence: ') 
    print(start_context)
    start_tokens = torch.tensor([tokenizer.encode(start_context)], device=device)
    print('###################################\n')  

    # Train MTA model
    print("Training MTA model...")
    train_mta_losses, val_mta_losses, total_examples_seen = train_model(
        model_mta, optimizer, 
        num_epochs, train_loader, val_loader, device,
        start_tokens, tokenizer, eval_freq=50, eval_batches=30) 

    # Save MTA pretraining loss plot
    epochs_seen = torch.linspace(0, num_epochs, len(train_mta_losses)) 
    examples_seen = torch.linspace(0, total_examples_seen, len(train_mta_losses))
    plot_dir = 'plots/mta_pretraining'
    plot_values(epochs_seen, examples_seen, train_mta_losses, val_mta_losses, 'loss', plot_dir)

    # Save model 
    torch.save(model_mta.state_dict(), 'model_mta.pth')
    print("Model weights saved to 'model_mta.pth'.")


if __name__ == '__main__': 
    main() 