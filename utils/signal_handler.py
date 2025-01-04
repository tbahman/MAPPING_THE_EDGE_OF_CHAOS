import signal
from utils.checkpoint import save_checkpoint

def handle_preemption(signum, frame, current_index, results, args_adls_output, checkpoint_ver):
    save_checkpoint(current_index, results, args_adls_output, checkpoint_ver)
    exit(0)

