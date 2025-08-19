import sys
import os
import shutil
import uuid
import subprocess as sp
from multiprocessing import Pool

import numpy as np


qc_rem = """
$rem
JOBTYPE force
SYMMETRY false
SYM_IGNORE true
QM_MM true
METHOD B3LYP
BASIS 6-31G*
SCF_CONVERGENCE 6 
SCF_ALGORITHM gdm_ls
MAX_SCF_CYCLES 2000
SKIP_CHARGE_SELF_INTERACT true
$end
"""


ELEMENTS = ["None", 'H', 'He',
            'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
            'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']


# richard is tired of debugging the mm_charge errors ...
def check_dir(path):
    try:
        os.path.exists(path)
    except FileNotFoundError:
        print(f"\n\tnot found: {path} !!!\n")
        base = os.path.dname(path)
        for ftype in ["inp", "out"]:
            command = f"cp {base}/qchem.{ftype} ."
            os.popen(command)
        sys.exit(1)
    


def get_element_symbols(atomic_number):
    try:
        return [ELEMENTS[i] for i in atomic_number]
    except TypeError:
        return ELEMENTS[atomic_number]


def get_qchem_input(qc_inp, qm_pos, qm_elem_num, mm_pos, mm_charge, charge, mult):
    assert len(qm_pos) == len(qm_elem_num)
    assert len(mm_pos) == len(mm_charge)
    qm_element = get_element_symbols(qm_elem_num)
    with open(qc_inp, 'w') as f:
        f.write("$molecule\n")
        f.write("%d %d\n" % (charge, mult))
        for i in range(len(qm_pos)):
            f.write("".join(["%-3s" % qm_element[i],
                             "%25.16f" % qm_pos[i, 0],
                             "%25.16f" % qm_pos[i, 1],
                             "%25.16f" % qm_pos[i, 2], "\n"]))
        f.write("$end" + "\n\n")

        f.write("$external_charges\n")
        for i in range(len(mm_pos)):
            f.write("".join(["%21.16f" % mm_pos[i, 0],
                             "%21.16f" % mm_pos[i, 1],
                             "%21.16f" % mm_pos[i, 2],
                             "%21.16f" % mm_charge[i], "\n"]))
        f.write("$end\n\n")

        f.write(qc_rem + "\n")


def get_qchem_force(qm_pos, qm_elem_num, mm_pos, mm_charge, charge, mult):
    cwd = "/tmp/van/run_" + str(uuid.uuid4())
    #cwd = "training_set"
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    get_qchem_input(cwd + "/qchem.inp", qm_pos, qm_elem_num, mm_pos, mm_charge, charge, mult)

    cmdline = f"cd {cwd};"
    cmdline += f"QCSCRATCH={cwd}; qchem -nt 4 qchem.inp qchem.out save > qchem_run.log"
    proc = sp.Popen(args=cmdline, shell=True)
    proc.wait()

#     energy = np.fromfile(cwd + "/save/99.0", dtype="f8", count=2)[1]
#     qm_grad = np.fromfile(cwd + "/save/131.0", dtype="f8").reshape(-1, 3)
#     mm_esp = np.fromfile(cwd + "/save/5001.0", dtype="f8", count=len(mm_charge))
#     mm_esp_grad = -np.fromfile(cwd + "/save/5002.0", dtype="f8", count=(len(mm_charge)*3)).reshape(-1, 3)
#     mm_esp = np.loadtxt(cwd + "/esp.dat", dtype=float)
#     mm_esp_grad = -np.loadtxt(cwd + "/efield.dat", max_rows=len(mm_esp), dtype=float)
    
    # richard 12/02/24
    QCFILES = ["save/99.0", "save/131.0", "save/5001.0", "save/5002.0", "esp.dat", "efield.dat"]
    for QCFILE in QCFILES:
        check_dir(f"{cwd}/{QCFILE}")

    energy = np.fromfile(cwd + "/save/99.0", dtype="f8", count=2)[1]
    qm_grad = np.fromfile(cwd + "/save/131.0", dtype="f8").reshape(-1, 3)
    mm_esp = np.fromfile(cwd + "/save/5001.0", dtype="f8", count=len(mm_charge))
    mm_esp_grad = -np.fromfile(cwd + "/save/5002.0", dtype="f8", count=(len(mm_charge)*3)).reshape(-1, 3)
    mm_esp = np.loadtxt(cwd + "/esp.dat", dtype=float)
    mm_esp_grad = -np.loadtxt(cwd + "/efield.dat", max_rows=len(mm_esp), dtype=float)

    shutil.rmtree(cwd)

    return energy, qm_grad, mm_esp, mm_esp_grad


def get_qchem_training_set(qm_coords, qm_elem_num, mm_coords, mm_charges, charge, mult):
    args = ((qm_pos, qm_elem_num, mm_pos, mm_charge, charge, mult) for qm_pos, mm_pos, mm_charge in zip(qm_coords, mm_coords, mm_charges))
    with Pool() as p:
        res = p.starmap(get_qchem_force, args)
    energy = np.zeros(len(qm_coords))
    qm_grad = np.zeros_like(qm_coords)
    mm_esp = np.zeros_like(mm_charges)
    mm_esp_grad = np.zeros_like(mm_coords)
    for i, r in enumerate(res):
        energy[i] = r[0]
        qm_grad[i] = r[1]
        mm_esp[i] = r[2]
        mm_esp_grad[i] = r[3]
    return energy, qm_grad, mm_esp, mm_esp_grad


if __name__ == "__main__":
    import os
    import sys
    import numpy as np

    #frame=str(sys.argv[1]) # v - should be %04d, bash `seq` easiest
    #base = f"{os.getcwd()}/{frame}" # v - for my
    #os.chdir(base)                  #     sanity
    #print(f"base: {base}")          #     ......

    QM_CHARGE = 0 # v - don't forget
    dirname="dft_forces"
    qm_coords = np.load(f"{dirname}/qm_coord.npy")
    mm_coords = np.load(f"{dirname}/mm_coord.npy")
    qm_elems = np.loadtxt("qm_elem.txt", dtype=int)
    mm_charges = np.load(f"{dirname}/mm_charge.npy")
    energy, qm_grad, mm_esp, mm_esp_grad  = get_qchem_training_set(qm_coords, qm_elems, mm_coords, mm_charges, QM_CHARGE, 1)
    out="train"
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/energy", energy)
    np.save(f"{out}/m_grad", np.array(qm_grad, dtype='float32'))
    np.save(f"{out}/mm_esp", np.array(mm_esp, dtype='float32'))
    np.save(f"{out}/mm_esp_grad", np.array(mm_esp_grad, dtype='float32'))
