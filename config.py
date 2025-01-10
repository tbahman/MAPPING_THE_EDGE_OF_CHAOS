
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--adls_output",
        help="path of the folder on ADLS where the training output files are stored",
        default=None
    )
    return parser.parse_args()



