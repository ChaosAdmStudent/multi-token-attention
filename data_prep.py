import torch
from torch.utils.data import DataLoader, Dataset 
import os 

import os

def prepare_babylm_dataset(data_folder, output_file):
    """
    Combines all .train files in the specified folder into a single file with <|endoftext|> as a separator.

    Args:
        data_folder (str): Path to the folder containing .train files.
        output_file (str): Path to the output file where combined content will be written.
    """
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file '{output_file}' already exists. Skipping merging.")
        return

    # Combine all .train files into a single file
    with open(output_file, "w") as outfile:
        for root, _, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".train"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as infile:
                        for line in infile:
                            outfile.write(line)  # Write line by line
                        outfile.write(" <|endoftext|> ")  # Add separator after each file
                    print('Finished combining file:', file)

    print(f"Combined files into '{output_file}'.")
    
class GPTDataset(Dataset): 

    def __init__(self, txt, tokenizer, max_length, stride):
        self.inputs= [] 
        self.outputs = [] 
        self.max_length = max_length 
        self.stride = stride 

        self.tokenized = tokenizer.encode(txt) 
        for i in range(0, len(self.tokenized) - max_length, stride): 
            x = self.tokenized[i:i+max_length] 
            y = self.tokenized[i+1: i+1+max_length]
            self.inputs.append(x) 
            self.outputs.append(y)

        self.inputs = torch.tensor(self.inputs) 
        self.outputs = torch.tensor(self.outputs) 

    def __len__(self): 
        return len(self.inputs) 

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]  
    
def create_dataloader(txt, tokenizer, max_length, stride, batch_size, shuffle, drop_last, num_workers): 
    dataset = GPTDataset(txt, tokenizer, max_length, stride) 
    return DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        drop_last = drop_last, 
        num_workers= num_workers
    )