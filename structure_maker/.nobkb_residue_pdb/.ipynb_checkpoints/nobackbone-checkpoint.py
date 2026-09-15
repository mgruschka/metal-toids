import mdtraj as md
import numpy as np
import sys

def remove_backbone(filename):
    mol = md.load(filename, standard_names=False)
    top = mol.top
    correct_indices = np.array(top.select('not name NL and not name CA and not name CLP and not name OL and not name HA1 and not name HA2 and not name HA'))
    onswitch = np.zeros(top.n_atoms, dtype=np.bool)
    onswitch[correct_indices] = 1
    for i in range(top.n_atoms - 1, -1, -1):
        if not onswitch[i]:
            top.delete_atom_by_index(i)
    new_traj = md.Trajectory(mol.xyz[0, onswitch], top)
    new_traj.save_pdb(filename)
    
remove_backbone(str(sys.argv[1]))
