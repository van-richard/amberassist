#!/home/van/miniforge3/bin/python
"""
metal_cv_builder.py
-------------------
Generate AMBER &rst distance restraints (cv.rst) for a metal coordination site.

Key inputs (Amber mask for metal)
=================================
Examples:
  --metal ":MG&:371"            - match residue name MG at resseq 371 (any chain)
  --metal ":371"                - match residue number 371 (any resname, any chain)
  --metal ":370-372,380&A"      - match residues 370..372 and 380 on chain A (any resname)

Mask grammar (supported): one or more colon-prefixed selectors joined by '&'.
Selectors can be:
  :RESNAME      (e.g., :MG, :ZN)
  :RESSEQSET    (e.g., :371, :370-372,380,400-402)
  :CHAIN        (e.g., :A)

Generalized residue donor
=========================
--res GLU --cutoff 3.0
  - Scans for all residues named GLU across the PDB
  - Picks a chemically relevant donor atom (OE1/OE2…) that is within cutoff
  - If multiple qualify, chooses the nearest to the metal

Also written
============
- metal—O3' (nearest O3'/O3*)
- metal—OP? (nearest OP1/OP2 or O1P/O2P)
- metal—<RES donor> (within cutoff)
- metal—3×water O (explicit via --waters or nearest three)

Defaults (edit below if desired)
================================
- O3'/OP?/RES donors: r2 = 2.0, r3 = 2.2, rk2 = rk3 = 50.0
- Waters:             r2 = 2.0, r3 = 2.0, rk2 = rk3 = 50.0
- All:                r1 = 0.0, r4 = 5.0
"""

import argparse
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

# -----------------------------
# PDB parsing and data classes
# -----------------------------

class Atom:
    """Minimal ATOM/HETATM representation from a PDB file."""
    __slots__ = (
        "serial","name","altloc","resname","chain","resseq","insertion",
        "x","y","z","element"
    )
    def __init__(
        self, serial:int, name:str, altloc:str, resname:str, chain:str,
        resseq:int, insertion:str, x:float, y:float, z:float, element:str
    ):
        self.serial = serial
        self.name = name.strip()
        self.altloc = altloc.strip() or ""
        self.resname = resname.strip()
        self.chain = chain.strip() or ""
        self.resseq = resseq
        self.insertion = insertion.strip() or ""
        self.x, self.y, self.z = x, y, z
        self.element = (element or "").strip()

    def coord(self) -> Tuple[float,float,float]:
        return (self.x, self.y, self.z)

    def resid_key(self) -> Tuple[str,str,int,str]:
        """Residue key: (chain, resname, resseq, insertion)."""
        return (self.chain, self.resname, self.resseq, self.insertion)


def parse_pdb(path: str) -> List[Atom]:
    """Parse ATOM/HETATM records from a PDB file using fixed columns."""
    atoms: List[Atom] = []
    with open(path, "r") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                serial = int(line[6:11])
            except ValueError:
                continue
            name = line[12:16]
            altloc = line[16:17]
            resname = line[17:20]
            chain = line[21:22]
            resseq_str = line[22:26].strip()
            try:
                resseq = int(resseq_str)
            except ValueError:
                continue
            insertion = line[26:27]
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            element = line[76:78].strip() if len(line) >= 78 else ""
            atoms.append(Atom(serial, name, altloc, resname, chain, resseq, insertion, x, y, z, element))
    return atoms


def distance(a: Atom, b: Atom) -> float:
    """Euclidean distance (Å)."""
    dx = a.x - b.x; dy = a.y - b.y; dz = a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

# -----------------------------
# Amber mask parsing (supports missing resname)
# -----------------------------

def _expand_resseq_item(token: str) -> List[int]:
    if '-' in token:
        lo, hi = token.split('-', 1)
        lo_i = int(lo); hi_i = int(hi)
        if hi_i < lo_i:
            lo_i, hi_i = hi_i, lo_i
        return list(range(lo_i, hi_i+1))
    return [int(token)]

def _expand_resseq_spec(spec: str) -> Set[int]:
    out: Set[int] = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        out.update(_expand_resseq_item(part))
    return out

def parse_amber_mask(mask: str) -> Tuple[Optional[str], Optional[Set[int]], Optional[str]]:
    """Parse masks like ':MG&:371', ':371', ':370-372,380&A'.
    Returns (resname or None, resseq_set or None, chain or None).
    """
    if not mask or not mask.startswith(':'):
        raise ValueError(f"Bad --metal mask '{mask}'. Expected like ':MG&:371' or ':371'.")

    segs = [seg.strip() for seg in mask.split('&') if seg.strip()]
    segs = [seg[1:] if seg.startswith(':') else seg for seg in segs]

    resname: Optional[str] = None
    resseqs: Optional[Set[int]] = None
    chain: Optional[str] = None

    for seg in segs:
        if seg.isalpha():
            if resname is None:
                resname = seg.upper()
            else:
                chain = seg
        elif any(ch.isdigit() for ch in seg):
            resseqs = _expand_resseq_spec(seg)
        else:
            chain = seg

    return resname, resseqs, chain

# -----------------------------
# Selection helpers
# -----------------------------

WATER_RESNAMES = {"HOH","WAT","SOL","TIP3","TIP3P","TIP4P","H2O"}
WATER_O_NAMES = {"O","OW","OH2"}

# Preferred donor atoms per residue
PREFERRED_DONORS: Dict[str, Tuple[str, ...]] = {
    "ASN": ("OD1",),
    "GLN": ("OE1",),
    "ASP": ("OD1","OD2"),
    "GLU": ("OE1","OE2"),
    "HIS": ("NE2","ND1"),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
    "LYS": ("NZ",),
    "ARG": ("NH1","NH2","NE"),
    "CYS": ("SG",),
}

# Element symbols commonly used for metals in PDBs
METAL_ELEMENTS = {
    "MG","MN","ZN","CA","NA","K","CU","CO","NI","FE","CD","HG","SR","BA","YB","PB","AG","PT","AU"
}


def filter_residue_atoms(atoms: List[Atom], resname: Optional[str], resseqs: Optional[Set[int]], chain: Optional[str]) -> List[Atom]:
    """Return atoms matching resname (if provided), residue numbers (if provided), and chain (if provided)."""
    out: List[Atom] = []
    for at in atoms:
        if resname is not None and at.resname.upper() != resname.upper():
            continue
        if resseqs is not None and at.resseq not in resseqs:
            continue
        if chain not in (None, "", at.chain):
            continue
        out.append(at)
    return out


def find_metal_atom_by_mask(atoms: List[Atom], resname: Optional[str], resseqs: Optional[Set[int]], chain: Optional[str]) -> Atom:
    """Find the metal atom from (optional) resname/resseqs/chain.

    Preference order:
      1) atom whose element is a known metal (METAL_ELEMENTS)
      2) if resname given: atom name == resname (e.g., 'MG')
      3) if resname given: atom element == resname
      4) lowest serial as fallback
    """
    candidates = filter_residue_atoms(atoms, resname, resseqs, chain)
    if not candidates:
        rs = (f" residues {sorted(resseqs)}" if resseqs else " (any residue id)")
        ch = (f" chain '{chain}'" if chain else "")
        rn = (resname or "any resname")
        raise ValueError(f"No atoms found for selection {rn}{rs}{ch}.")

    def score(a: Atom) -> Tuple[int,int,int,int]:
        name_u = a.name.strip().upper()
        elem_u = a.element.strip().upper()
        # Prefer atoms that look like metals
        is_metal_elem = 0 if elem_u in METAL_ELEMENTS else 1
        name_match = 0 if (resname and name_u == resname.upper()) else 1
        elem_match = 0 if (resname and elem_u == resname.upper()) else 1
        return (is_metal_elem, name_match, elem_match, a.serial)

    return sorted(candidates, key=score)[0]


def find_nearest_by_name(atoms: List[Atom], metal: Atom, name_targets: Iterable[str]) -> Atom:
    """Find nearest atom whose name is in the target list (case-insensitive)."""
    name_set = {n.upper() for n in name_targets}
    best = None
    best_d = float("inf")
    for a in atoms:
        if a.altloc not in ("", "A"):
            continue
        if a.name.strip().upper() in name_set:
            d = distance(metal, a)
            if d < best_d:
                best, best_d = a, d
    if best is None:
        raise ValueError(f"Could not find any atoms with names in {sorted(name_set)}.")
    return best


def find_nearest_OP(atoms: List[Atom], metal: Atom) -> Atom:
    return find_nearest_by_name(atoms, metal, ["OP1","OP2","O1P","O2P"])


def find_nearest_O3prime(atoms: List[Atom], metal: Atom) -> Atom:
    return find_nearest_by_name(atoms, metal, ["O3'", "O3*"])


def residues_by_name(atoms: List[Atom], resname: str) -> Dict[Tuple[str,str,int,str], List[Atom]]:
    by_res = defaultdict(list)
    rn = resname.upper()
    for a in atoms:
        if a.resname.upper() == rn:
            by_res[a.resid_key()].append(a)
    return by_res


def choose_residue_donor_atom(res_atoms: List[Atom], resname: str, metal: Atom) -> Optional[Atom]:
    name_order = PREFERRED_DONORS.get(resname.upper(), ())
    preferred = [a for a in res_atoms if a.name.strip().upper() in set(name_order)]
    candidates = preferred if preferred else [a for a in res_atoms if a.element.upper() in {"O","N"}]
    if not candidates:
        return None
    return min(candidates, key=lambda a: distance(metal, a))


def find_residue_atom_by_resname_within_cutoff(atoms: List[Atom], resname: str, metal: Atom, cutoff: float) -> Atom:
    by_res = residues_by_name(atoms, resname)
    if not by_res:
        raise ValueError(f"No residues named '{resname}' found in the PDB.")
    best_atom = None
    best_dist = float("inf")
    for _, alist in by_res.items():
        donor = choose_residue_donor_atom(alist, resname, metal)
        if donor is None:
            continue
        d = distance(metal, donor)
        if d <= cutoff and d < best_dist:
            best_atom, best_dist = donor, d
    if best_atom is None:
        raise ValueError(f"No coordinating atom from residues named '{resname}' within cutoff {cutoff:.2f} Å of the metal.")
    return best_atom


def list_water_O_atoms(atoms: List[Atom], chain: Optional[str] = None) -> List[Tuple[Tuple[str,str,int,str], Atom]]:
    res_to_O: List[Tuple[Tuple[str,str,int,str], Atom]] = []
    by_res = defaultdict(list)
    for a in atoms:
        if a.resname in WATER_RESNAMES and (chain is None or chain == "" or a.chain == chain):
            by_res[a.resid_key()].append(a)
    for resid_key, alist in by_res.items():
        o_candidates = [a for a in alist if a.name.strip().upper() in WATER_O_NAMES]
        if not o_candidates:
            o_candidates = [a for a in alist if a.element.strip().upper() == "O"]
        if o_candidates:
            o_atom = sorted(o_candidates, key=lambda x: x.serial)[0]
            res_to_O.append((resid_key, o_atom))
    return res_to_O


def pick_water_Os(atoms: List[Atom], metal: Atom, waters_csv: Optional[str], chain: Optional[str]) -> List[Atom]:
    if waters_csv:
        wanted = [int(x.strip()) for x in waters_csv.split(",") if x.strip()]
        picked: List[Atom] = []
        for resid in wanted:
            candidates = []
            for (_, _, resseq, _), oatom in list_water_O_atoms(atoms, chain):
                if resseq == resid:
                    candidates.append(oatom)
            if candidates:
                picked.append(min(candidates, key=lambda a: distance(metal, a)))
        if len(picked) < 3:
            already = {a.serial for a in picked}
            remaining = [(distance(metal, o), o) for _, o in list_water_O_atoms(atoms, chain) if o.serial not in already]
            remaining.sort(key=lambda t: t[0])
            for _, o in remaining:
                if len(picked) >= 3:
                    break
                picked.append(o)
        return picked[:3]
    else:
        water_list = list_water_O_atoms(atoms, chain)
        with_d = [(distance(metal, o), o) for _, o in water_list]
        with_d.sort(key=lambda t: t[0])
        return [o for _, o in with_d[:3]]

# -----------------------------
# Restraint writing
# -----------------------------

def rst_block(label: str, i: int, j: int, r2: float, r3: float) -> str:
    r1 = 0.0
    r4 = 5.0
    rk2 = 50.0
    rk3 = 50.0
    return (
        f"# {label}\n"
        " &rst\n"
        f"  iat={i},{j}\n"
        f"  r1={r1},r2={r2},r3={r3},r4={r4},\n"
        f"  rk2={rk2},rk3={rk3},\n"
        " &end\n\n"
    )


def build_cv_rst(
    atoms: List[Atom],
    metal_mask: str,
    resname: str,
    cutoff: float,
    waters_csv: Optional[str],
    water_chain: Optional[str]
) -> str:
    m_resname, m_resseqs, m_chain = parse_amber_mask(metal_mask)
    metal = find_metal_atom_by_mask(atoms, m_resname, m_resseqs, m_chain)

    o3p = find_nearest_O3prime(atoms, metal)
    op  = find_nearest_OP(atoms, metal)
    donor = find_residue_atom_by_resname_within_cutoff(atoms, resname, metal, cutoff)
    waters = pick_water_Os(atoms, metal, waters_csv, water_chain)

    blocks: List[str] = []
    metal_label = metal.element or (m_resname or metal.resname)
    blocks.append(rst_block(f"{metal_label} — O3'", metal.serial, o3p.serial, 2.0, 2.2))
    blocks.append(rst_block(f"{metal_label} — {op.name} (nearest)", metal.serial, op.serial, 2.0, 2.2))
    blocks.append(rst_block(
        f"{metal_label} — {donor.resname} {donor.resseq} {donor.name} (≤ {cutoff:.2f} Å)",
        metal.serial, donor.serial, 2.0, 2.2
    ))
    for k, w in enumerate(waters, start=1):
        blocks.append(rst_block(f"{metal_label} — water#{k} {w.resname} {w.resseq} {w.name}",
                                metal.serial, w.serial, 2.0, 2.0))
    return "".join(blocks)

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Build cv.rst restraints for a metal coordination site.")
    parser.add_argument("--pdb", default="step3_pbcsetup.pdb", help="Input PDB file")
    parser.add_argument("--out", default="cv.rst", help="Output restraints file (default: cv.rst)")

    parser.add_argument("--metal", required=True,
                        help="Amber mask-style metal selector, e.g. ':MG&:371', ':371', or ':370-372,380&A'.")

    parser.add_argument("--res", required=True,
                        help="Residue NAME (e.g., ASN, GLN, ASP, GLU, HIS). The script finds a donor atom "
                             "from any such residue within --cutoff Å of the metal.")
    parser.add_argument("--cutoff", type=float, default=2.5,
                        help="Maximum distance (Å) from metal to the chosen donor atom for --res.")

    parser.add_argument("--waters", default="",
                        help="Comma-separated list of water residue numbers (e.g., 398,399,400). "
                             "If omitted, auto-select the three closest water O atoms.")
    parser.add_argument("--water-chain", default="", help="Chain ID for waters (optional).")

    args = parser.parse_args()
    atoms = parse_pdb(args.pdb)

    content = build_cv_rst(
        atoms=atoms,
        metal_mask=args.metal,
        resname=args.res,
        cutoff=args.cutoff,
        waters_csv=(args.waters or None),
        water_chain=args.water_chain
    )

    with open(args.out, "w") as fh:
        fh.write(content)

    print(f"Wrote restraints to: {args.out}")


if __name__ == "__main__":
    main()
