import os
import sys
import numpy as np
from glob import glob

n_windows = 42
n_frames = 100

dirname=str(sys.argv[1])
filename=str(sys.argv[2])

os.makedirs(dirname, exist_ok=True)

fnames = []

for window in range(n_windows):
    fname = sorted(glob(f'../{window:02d}/dft_forces/*/{filename}.npy'))

    print(window, len(fname))
    fnames.append(fname)
    
data = [np.load(fnames[window][frame]) for frame in range(n_frames) for window in range(n_windows)]
datas = np.concatenate(data)

np.save(f'{dirname}/{filename}', datas)

