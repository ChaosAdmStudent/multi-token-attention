import torch 
import torch.nn as nn
from torch.utils.data import Sampler
import numpy as np
import matplotlib.pyplot as plt 
import os 
import random

def split_txt_train_test_val(raw_text, train_frac=0.7, val_frac=0.2): 
    train_idx = int(train_frac * len(raw_text)) 
    val_idx = train_idx +  int(val_frac * len(raw_text)) 

    train_txt = raw_text[:train_idx] 
    val_txt = raw_text[train_idx: val_idx] 
    test_txt = raw_text[val_idx: ] 

    return train_txt, val_txt, test_txt

def plot_values(epochs_seen, examples_seen, train_values, val_values, label:str, plot_dir): 
    fig, ax1 = plt.subplots(figsize=(5,3)) 

    # Plot Train values 
    ax1.plot(epochs_seen, train_values, label=f"Training {label}") 
    ax1.plot(epochs_seen, val_values, label=f"Validation {label}", linestyle='-.') 
    ax1.set_xlabel("Epochs seen") 
    ax1.set_ylabel(label.capitalize()) 
    ax1.legend()

    # Plot Val values
    ax2 = ax1.twiny() 
    ax2.plot(examples_seen, train_values, alpha=0) # Invisible/transparent plot for aligning ticks for 2 X-axis
    ax2.set_xlabel("Examples seen") 

    fig.tight_layout() 

    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/model_{label}.png', bbox_inches='tight') 

def generate_out_text(max_new_tokens, model, input_token_embeddings: torch.Tensor, tokenizer, context_length, device, temperature=0.0, topk=None, eos_id=50256): 

    assert len(input_token_embeddings.shape) == 2, "Input token embeddings must be shape (B,N)"
    B, _ = input_token_embeddings.shape
 
    out_token_embeddings = input_token_embeddings # (B,N)    

    model.to(device) 
    input_token_embeddings = input_token_embeddings.to(device) 
    finished = torch.zeros(B, device=device)

    # Generate output tokens 
    for i in range(max_new_tokens): 
        if finished.all(): # If all have EOS tokens 
            break 

        with torch.no_grad(): 
            logits: torch.Tensor = model(out_token_embeddings[:, -context_length: ]) # (B,N,vocab_size)
            last_token_logits = logits[:, -1, :] # (B, vocab_size)

            if topk is not None: 
                top_logits, top_pos = torch.topk(last_token_logits, topk, dim=-1) 
                last_token_logits = torch.where(
                    condition=last_token_logits < top_logits[-1], 
                    input=  torch.tensor(-torch.inf).to(last_token_logits.device),
                    other=last_token_logits
                )

            if temperature > 0: 
                last_token_logits /= temperature 
                next_token_probas = torch.softmax(last_token_logits, dim=-1)
                next_token_ids = torch.multinomial(next_token_probas, 1) # (B,1)

            else: 
                next_token_ids = torch.argmax(last_token_logits, dim=-1, keepdim=True) # (B,1) 

            # Replace generated token_ids for examples in the batch which already finished (encountered EOS) with EOS token: 
            next_token_ids[finished] = eos_id 

            # Update finished 
            newly_finished = next_token_ids.squeeze(1) == eos_id # (B,) 
            finished = finished | newly_finished

            out_token_embeddings = torch.cat((out_token_embeddings, next_token_ids), dim=1) # (B, N+1) 

    # Generate text 
    out_texts = [] 
    for i in range(out_token_embeddings.shape[0]): 
        tokens = out_token_embeddings[i].tolist() 
        if eos_id in tokens: 
            eos_idx = tokens.index(eos_id) 
            tokens = tokens[:eos_idx] 
        
        text = tokenizer.decode(tokens) 
        out_texts.append(text) 

    return out_texts

def collate_fn_dynamic_padding(batch, pad_id = 50256, ignore_index = -100, allowed_max_length = None, mask_instruction = False): 
    max_batch_length = max([len(item[0] + item[1] + 1 for item in batch)]) 
    inputs_lst = [] 
    outputs_lst = [] 

    for item in batch: 
        instruction, response = item 
        new_item = instruction + response 
        padded = new_item + [pad_id] * (max_batch_length - len(new_item))
        input = padded[:-1] 
        output = padded[1:] 

        # Mask extra pad tokens with ignore_index so that loss can ignore those later
        mask = output == pad_id 
        indices = torch.nonzero(mask).squeeze() 
        if indices.numel() > 1: 
            output[mask[1:]] = ignore_index
        

        # Use max_length if specified 
        if allowed_max_length is not None: 
            input = input[:allowed_max_length] 
            output = output[:allowed_max_length] 
        
        # Mask instruction tokens with ignore_index if specified
        if mask_instruction: 
            output[:len(instruction)-1] = ignore_index
        
        inputs_lst.append(input) 
        outputs_lst.append(output) 
    
    inputs_tensor: torch.Tensor = torch.stack(inputs_lst) 
    outputs_tensor: torch.Tensor = torch.stack(outputs_lst) 
    
    return inputs_tensor, outputs_tensor

class BucketingSampler(Sampler): 
    
    def __init__(self, seq_lens): 
        pass 

    def __len__(self): 
        pass 

    def __iter__(self):
        # Returns indices of the batches 
        pass

class GELU(nn.Module): 
    def __init__(self): 
        super(GELU, self).__init__() 

    def forward(self, x): 
        return (0.5 * x * (1+ torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.044715 * torch.pow(x,3)))
        )
        ) 

def assign(left: torch.Tensor, right: torch.Tensor): 
    if left.shape != right.shape: 
        raise ValueError(f"Shape mismatch. Left shape: {left.shape}, Right shape: {right.shape}") 
    else: 
        return nn.Parameter(torch.tensor(right)) 
    
def load_weights_into_gpt(gpt, params):
    """
    gpt: GPT model architecture
    params: Pretrained gpt model open weights from OpenAI
    """

    gpt.pos_emb_layer.weight = assign(gpt.pos_emb_layer.weight, params['wpe'])
    gpt.token_emb_layer.weight = assign(gpt.token_emb_layer.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.proj_out.weight = assign(
            gpt.trf_blocks[b].att.proj_out.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.proj_out.bias = assign(
            gpt.trf_blocks[b].att.proj_out.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].layer_norm1.scale = assign(
            gpt.trf_blocks[b].layer_norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].layer_norm1.shift = assign(
            gpt.trf_blocks[b].layer_norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].layer_norm2.scale = assign(
            gpt.trf_blocks[b].layer_norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].layer_norm2.shift = assign(
            gpt.trf_blocks[b].layer_norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True