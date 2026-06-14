import numpy as np
import os
from pathlib import Path

output_dir = Path(os.environ.get("EDA_OUTPUT_DIR", "eda"))
f = np.load(output_dir / 'lj_prot_forces.npy')
force = f.sum(axis=3)

np.save(output_dir / 'lj_prot_forces_sum.npy', force)
