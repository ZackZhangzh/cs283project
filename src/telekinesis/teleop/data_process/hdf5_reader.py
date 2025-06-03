import os 
import sys

import h5py
from pathlib import Path




def main():
    dataset_name = "test.hdf5"

    
    if Path(dataset_name).exists():
        with h5py.File(dataset_name, 'r') as f:
            episode_idx = len(f['data'].keys())
            

if __name__ == "__main__":
    main()