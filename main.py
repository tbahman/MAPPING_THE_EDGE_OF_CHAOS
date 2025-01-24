import os
import json
from datetime import datetime
import numpy as np
from itertools import product
from utils.preprocessing import clean_gutenberg_text, build_vocab, tokenize, create_dataset
from utils.signal_handler import register_signal_handler
from model.training import (
    train_and_evaluate_learning_rates,
    save_results_to_json
)

def main(args):
    global args_adls_output
    global checkpoint_ver

    args_adls_output = args.adls_output
    register_signal_handler(args_adls_output, current_index=0, results={}, checkpoint_ver=1)

    _n_ = 5
    seq_length = 64
    num_values_t = 20
    num_values_fc = 150
    fc_lr_range = (0.102002212, 0.1020022135)
    percision_ = 0.00000000001
    a_ = round(0.025000224 + (_n_ - 1) * percision_ * num_values_t, 10)
    b_ = round(0.025000224 + _n_ * percision_ * num_values_t, 10)
    t_lr_range = (a_, b_)
    checkpoint_ver = int(8000 + _n_)
    
    print(t_lr_range, fc_lr_range, num_values_t, num_values_fc, checkpoint_ver)

    file_in_path = 'gutenberg_100.txt'
    with open(file_in_path, 'r', encoding='utf-8') as file:
        text = file.read()

    cleaned_text = clean_gutenberg_text(text)
    char2idx, idx2char = build_vocab(cleaned_text)
    tokens = tokenize(cleaned_text, char2idx)
    inputs, targets = create_dataset(tokens, seq_length)
    test_size = int(len(inputs) * 0.96)
    train_size = int(len(inputs) * 0.2)
    X_train = inputs[:train_size]
    y_train = targets[:train_size]
    X_val = inputs[test_size:]
    y_val = targets[test_size:]
    train_ds = {'X': X_train, 'y': y_train}
    val_ds = {'X': X_val, 'y': y_val}

    t_lr_values = np.linspace(t_lr_range[0], t_lr_range[1], num=num_values_t)
    fc_lr_values = np.linspace(fc_lr_range[0], fc_lr_range[1], num=num_values_fc)
    learning_rate_pairs = list(product(t_lr_values, fc_lr_values))
