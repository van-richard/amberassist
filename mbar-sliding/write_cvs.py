import sys
import numpy as np
from glob import glob

steprep=sys.argv[1]
step = steprep.split('.')[0]
rep = steprep.split('.')[1]

n_windows = int(sys.argv[2]) # n_windows

lengths = []
val_kn = []
for i in range(n_windows):
    fnames = sorted(glob(f'../{i:02d}/{step}.{rep}_equilibration.cv'))

    if len(fnames) == 0:
        raise FileNotFoundError(f"No CV files found for window {i:02d}: ../{i:02d}/{step}.{rep}_equilibration.cv")

    arrays = []
    for fname in fnames[:]:
        f = np.loadtxt(fname, usecols=1)[:100:]
        arrays.append(f)
    
    val_kn.append(np.concatenate(arrays))
    lengths.append(len(np.concatenate(arrays)))


if len(set(lengths)) != 1:
    raise ValueError("Not all windows have the same number of samples. "
                     "This will produce an object array and break MBAR. "
                     "Check rep wildcard and missing/extra files.")


val_kn = np.stack(val_kn, axis=0).astype(float)
np.save('cvs', val_kn)

print("Final val_kn shape:", val_kn.shape, "dtype:", val_kn.dtype)

