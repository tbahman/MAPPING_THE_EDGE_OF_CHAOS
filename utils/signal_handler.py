import signal
from utils.checkpoint import save_checkpoint

def handle_preemption(signum, frame, current_index, results, args_adls_output, checkpoint_ver):
    save_checkpoint(current_index, results, args_adls_output, checkpoint_ver)
    exit(0)

def register_signal_handler(args_adls_output):
    signal.signal(signal.SIGTERM, handle_preemption)

