import pandas as pd
from pathlib import Path
import sys

# === Define input residue data ===
# Each tuple contains: (category label, residue name, residue index or list of indices)
residue_data = [
    ("metal", "MG", "371"),
    ("dna3term", "DA", "13"),
    ("dna5term", "DG", "14"),
    ("nuc", "WAT", "729"),
    ("base", "HIS", "303"),
    ("rescoord", "ASN", "324"),
    ("watcoord", "WAT", "601,611,612")
]

# === Create initial DataFrame ===
columns = ["category", "resname", "residx"]
df = pd.DataFrame(residue_data, columns=columns)

# === Expand comma-separated residue indices into individual rows ===
df_expanded = df.assign(residx=df["residx"].str.split(',')).explode("residx")
df_expanded["residx"] = df_expanded["residx"].str.strip()

# === Derive filename based on script name ===
script_name = Path(__file__).stem if '__file__' in globals() else "res_init"
outfile = Path(f"{script_name}.tsv")
df_expanded.to_csv(outfile, sep='\t', index=False)

print(f"Residue definition file saved as: {outfile.resolve()}")
