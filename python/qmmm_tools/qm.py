# qm.py

import glob
import os
import re


def get_qm_residues(directory='../00', pattern='step5.00*mdin'):
    mdin_files = glob.glob(os.path.join(directory, pattern))

    if len(mdin_files) != 1:
        raise RuntimeError(
            f"Expected exactly one mdin file in {directory}, found {len(mdin_files)}"
        )

    mdin_file = mdin_files[0]

    with open(mdin_file, 'r') as f:
        for line in f:
            if 'qmmask' in line:
                match = re.search(r"qmmask\s*=\s*'(.*?)'", line)
                if match:
                    qmmask = match.group(1)
                    break
        else:
            raise RuntimeError("qmmask not found")

    residue_blocks = re.findall(r':([0-9,\-]+)', qmmask)

    residues = []

    for block in residue_blocks:
        parts = block.split(',')
        for part in parts:
            if '-' in part:
                start, end = part.split('-')
                residues.extend(range(int(start), int(end) + 1))
            else:
                residues.append(int(part))

    return list(dict.fromkeys(residues))

