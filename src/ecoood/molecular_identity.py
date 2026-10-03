from __future__ import annotations

from rdkit import Chem, rdBase


POLYMER_GROUP_TYPES = frozenset({"SRU", "MON", "COP", "MER", "CRO", "GRA", "MOD", "ANY"})


def fixed_composition_molecule(value: object) -> Chem.Mol | None:
    """Parse a carbon-containing molecular input with fully specified composition."""
    if not isinstance(value, str) or not value.strip():
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value.strip())
    if molecule is None:
        return None
    atoms = list(molecule.GetAtoms())
    if not any(atom.GetAtomicNum() == 6 for atom in atoms):
        return None
    if any(atom.GetAtomicNum() == 0 or atom.HasQuery() for atom in atoms):
        return None
    if any(bond.HasQuery() for bond in molecule.GetBonds()):
        return None
    # Link nodes and polymer groups describe variable repetition, not a fixed molecule.
    if molecule.HasProp("_molLinkNodes"):
        return None
    if any(
        group.GetProp("TYPE") in POLYMER_GROUP_TYPES
        for group in Chem.GetMolSubstanceGroups(molecule)
        if group.HasProp("TYPE")
    ):
        return None
    return molecule
