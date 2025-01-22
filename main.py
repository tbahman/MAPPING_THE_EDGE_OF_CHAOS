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
