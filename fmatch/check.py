import os
import sys

def check_dir(path):
    try:
        os.listdir(path)
    except FileNotFoundError:
        print(f"\n\tnot found: {path} !!!\n")
        sys.exit(1)
    
    
