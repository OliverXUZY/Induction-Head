# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import os
import json 
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
# warnings.simplefilter("ignore")
print(torch.__version__)
import argparse
import yaml
from torch.optim.lr_scheduler import LambdaLR
from functools import partial

from src.config import Config 
from src.utils import fix_random_seed, create_folder, ensure_path
from src.data.gen_simple import gen_simple_data
from src.data.load_data import load_dataset_for_training, create_dataset
from src.model.simpleTF import TFModel, simpleT, simple2layerT
from src.train import train
from src.utils import plot_err_curve, Timer, time_str
from src.model import tinyGPT2LMHeadModel
from src.optimization import _get_cosine_schedule_with_warmup_lr_lambda
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


import wandb

wandb.init(project="tinyGPT2", entity="zhuoyan")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="save path for model ckpt")
    parser.add_argument("--config", type=str, default=None, help="config file to load from")

    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    
    parser.add_argument("--pattern", type=str, default='random',
                        help="pattern for generating data")
    
    return parser.parse_args()


def main():

    args = parse_args()
    
    
    seed = 2024
    fix_random_seed(seed)

    # Load tokenizer and model
    model_name = "gpt2"
    config = GPT2Config.from_pretrained("gpt2")
    config.n_head = 1
    config.n_layer = 2

    # if getattr(config, "tie_word_embeddings", True):
    #     print("tie!!")

    model = tinyGPT2LMHeadModel(config = config)

    gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Copy the embedding weights from the pre-trained model to your custom model
    model.transformer.wte.weight = gpt2.transformer.wte.weight
    model.transformer.wpe.weight = gpt2.transformer.wpe.weight

    model.tie_weights()  
    # Access the weights of the input embedding layer and the output linear layer
    embedding_weights = model.transformer.wte.weight
    output_weights = model.lm_head.weight


    # Check if they are the same (pointing to the same memory location)
    is_tied = embedding_weights.data_ptr() == output_weights.data_ptr()

    print(f"Weight tying implemented: {is_tied}")

    #freeze embedding parameters in the model
    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     if "wte" in name or "wpe" in name:
    #         param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")
        else:
            print(f"Frozen: {name}")


    
    ###### Load and preprocess the dataset
    block_size = 512  # Adjust based on your GPU memory
    data = "random_gen"
    if data == "openwebtext":
        train_set = load_dataset_for_training(tokenizer, block_size)

    elif data == "random_gen":
        vocab = torch.arange(5).type(torch.LongTensor)
        train_set = create_dataset(tokenizer=tokenizer, vocab = vocab, max_seq_len = block_size, sample_size = 2000, pattern="random")
        
    else:
        raise NotImplementedError

    # Data collator used for dynamic padding
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ##### train 
    ### set up optimizer

    # Prepare custom optimizer and scheduler
    def get_optimizer_and_scheduler(model, training_args):
        parameters = []
        # store params & learning rates
        for idx, (name, param) in enumerate(model.named_parameters()):
            # append layer parameters
            if "attn" in name:
                decay = 5e-4
            else:
                decay = 0
            parameters += [{'params': [pa for na, pa in model.named_parameters() if na == name and pa.requires_grad],
                            'lr':     training_args.learning_rate,
                            'weight_decay': decay}]
        
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_args.max_steps//4)

        # Set up the cosine scheduler with warmup
        num_warmup_steps = 10  # You need to define this in your training_args
        num_training_steps = training_args.max_steps
        num_cycles = 0.5  # This is the default value for half a cosine wave

        lr_lambda = partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )

        scheduler = LambdaLR(optimizer, lr_lambda)

        return optimizer, scheduler
    

    wandb.login()
    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned-openwebtext",
        overwrite_output_dir=True,
        max_steps=1000,  # Adjust number of epochs based on your needs
        per_device_train_batch_size=2,  # Adjust batch size based on your GPU
        save_steps=10_000,
        logging_steps=1,
        save_total_limit=2,
        learning_rate= 0.001/4,
        prediction_loss_only=True,
    )


    # Initialize Trainer
    print("==")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        optimizers=get_optimizer_and_scheduler(model, training_args),
    )

    # Train the model
    trainer.train()

    model.save_pretrained('model_output')


    # Close the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()



    