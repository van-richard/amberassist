# rc.py

import re


def generate_rcs_from_cv(cv_file):

    with open(cv_file, 'r') as f:
        for line in f:
            if 'iat=' in line:
                match = re.search(r'iat\s*=\s*([0-9,]+)', line)
                if not match:
                    raise RuntimeError("Could not parse iat line")

                iat_values = match.group(1).strip(',').split(',')
                iat_values = [int(x) for x in iat_values]
                break
        else:
            raise RuntimeError("iat= line not found")

    if len(iat_values) != 4:
        raise RuntimeError("Expected exactly 4 atom indices in iat=")

    i1, i2, i3, i4 = iat_values

    return [
        f'@{i1} @{i2}',
        f'@{i3} @{i4}'
    ]


def get_atom_index(top, resid, atom_name):

    selection = top.select(f':{resid}@{atom_name}')

    if len(selection) != 1:
        raise RuntimeError(
            f"Could not uniquely find atom {atom_name} in residue {resid}"
        )

    return selection[0] + 1


def build_extra_rcs(top, extra_pairs):

    rcs_extra = []

    for (res1, atom1), (res2, atom2) in extra_pairs:

        idx1 = get_atom_index(top, res1, atom1)
        idx2 = get_atom_index(top, res2, atom2)

        rcs_extra.append(f'@{idx1} @{idx2}')

    return rcs_extra


def build_rclabels(top, rcs):

    rclabels = []

    for pair in rcs:

        atoms = pair.replace('@', '').split()
        idx1 = int(atoms[0]) - 1
        idx2 = int(atoms[1]) - 1

        atom1 = top.atom(idx1)
        atom2 = top.atom(idx2)

        label = f"{atom1.resname}({atom1.name})–{atom2.resname}({atom2.name})"
        rclabels.append(label)

    return rclabels

