import pandas as pd
from pathlib import Path
import pytraj as pt

# === Config ===
pdb_path = Path("../input/prod1.pdb")
residue_file = Path("res_init.tsv")

# === Load topology ===
traj = pt.load(str(pdb_path))
top = traj.top

# === Build PDB serial number lookup ===
pdb_atom_serials = {}
for line in pdb_path.read_text().splitlines():
    if line.startswith("ATOM") or line.startswith("HETATM"):
        serial = int(line[6:11].strip())
        name = line[12:16].strip()
        resname = line[17:20].strip()
        resid = int(line[22:26].strip())
        key = (resname, resid, name)
        pdb_atom_serials[key] = serial

# === Load residue definition ===
df_input = pd.read_csv(residue_file, sep='\t')
df_input["residx"] = df_input["residx"].astype(int)

# === Collect atom records ===
records = []
for _, row in df_input.iterrows():
    category, resname, residx = row["category"], row["resname"], row["residx"]

    if residx >= top.n_residues:
        print(f"[Warning] Residue index {residx} out of range. Skipping.")
        continue

    for atom_idx in pt.select(f":{residx}", top):
        atom = top.atom(atom_idx)
        key = (atom.resname, atom.resid + 1, atom.name)
        pdb_serial = pdb_atom_serials.get(key, "NA")
        records.append({
            "category": category,
            "resname": atom.resname,
            "resid": atom.resid + 1,
            "atom_name": atom.name,
            "atom_idx": pdb_serial + 1,
            "mask": f":{atom.resid}@{atom.name}"
        })

# === Output as multi-index TSV ===
df_atoms = pd.DataFrame(records)
df_atoms.set_index(["category", "resname", "resid", "atom_name"], inplace=True)

outfile = Path("res_init2table.tsv")
df_atoms.to_csv(outfile, sep='\t')
print(f"Saved: {outfile.resolve()}")