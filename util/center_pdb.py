import argparse
import requests
import os

def download_pdb(pdb_id):
	url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
	response = requests.get(url)
	if response.status_code != 200:
		raise ValueError(f"Could not retrieve PDB ID {pdb_id} from RCSB")
	return response.text

def read_local_pdb(filename):
	with open(filename, "r") as f:
		return f.read()

def parse_coordinates(pdb_lines):
	coords = []
	atom_lines = []
	for line in pdb_lines:
		if line.startswith("ATOM") or line.startswith("HETATM"):
			x = float(line[30:38])
			y = float(line[38:46])
			z = float(line[46:54])
			coords.append((x, y, z))
			atom_lines.append(line)
	return coords, atom_lines

def compute_centroid(coords):
	n = len(coords)
	cx = sum(x for x, _, _ in coords) / n
	cy = sum(y for _, y, _ in coords) / n
	cz = sum(z for _, _, z in coords) / n
	return (cx, cy, cz)

def shift_coordinates(atom_lines, centroid):
	cx, cy, cz = centroid
	shifted_lines = []
	for line in atom_lines:
		x = float(line[30:38]) - cx
		y = float(line[38:46]) - cy
		z = float(line[46:54]) - cz
		new_line = (line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:])
		shifted_lines.append(new_line)
	return shifted_lines

def process_pdb(pdb_text, output_prefix):
	pdb_lines = pdb_text.splitlines()
	coords, atom_lines = parse_coordinates(pdb_lines)
	centroid = compute_centroid(coords)
	shifted_atoms = shift_coordinates(atom_lines, centroid)

	# preserve all lines
	shifted_pdb = []
	atom_index = 0
	for line in pdb_lines:
		if line.startswith("ATOM") or line.startswith("HETATM"):
			shifted_pdb.append(shifted_atoms[atom_index])
			atom_index += 1
		else:
			shifted_pdb.append(line)

	output_file = f"{output_prefix}_centered.pdb"
	with open(output_file, "w") as f:
		f.write("\n".join(shifted_pdb) + "\n")
	print(f"Centered PDB written to {output_file}")

def main():
	parser = argparse.ArgumentParser(description="Center PDB coordinates at origin.")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("-pdb", "--pdb", help="Path to a local PDB file")
	group.add_argument("-rcsb", "--rcsb", help="PDB ID to download from RCSB")
	args = parser.parse_args()

	if args.pdb:
		pdb_text = read_local_pdb(args.pdb)
		output_prefix = os.path.splitext(os.path.basename(args.pdb))[0]
	else:
		pdb_id = args.rcsb.upper()
		pdb_text = download_pdb(pdb_id)
		output_prefix = pdb_id

	process_pdb(pdb_text, output_prefix)

if __name__ == "__main__":
	main()

