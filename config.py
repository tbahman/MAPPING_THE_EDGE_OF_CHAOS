


import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--adls_output", default=None, help="Path of the folder on ADLS for output files"
)
ARGS = parser.parse_args()



