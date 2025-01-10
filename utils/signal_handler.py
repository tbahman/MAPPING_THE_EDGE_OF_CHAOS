import signal
import sys
from utils.checkpoint import save_checkpoint

def handle_preemption(signum, frame, args_adls_output, current_index, results, checkpoint_ver):
    print("Preemption signal received, saving checkpoint...")
    save_checkpoint(current_index, results, args_adls_output, checkpoint=checkpoint_ver)
    sys.exit(0)

def register_signal_handler(args_adls_output, current_index, results, checkpoint_ver):
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: handle_preemption(signum, frame, args_adls_output, current_index, results, checkpoint_ver)
    )
