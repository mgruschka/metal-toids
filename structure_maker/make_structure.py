import numpy as np
import copy
import mdtraj as md
import os
import argparse
import subprocess
import sys
from Bio.PDB import PDBIO
from Bio.PDB.vectors import Vector, rotaxis, calc_dihedral, rotmat
from Bio.SVDSuperimposer import SVDSuperimposer
from PeptideBuilder import Geometry
import PeptideBuilder

from collections import defaultdict
import itertools
import json

MINIMA_FILENAMES = ['AD-C', 'AD+C', 'A-C', 'A+C', 'C7B-C', 'C7B+C', 'AD-T', 'AD+T', 'A-T', 'A+T', 'C7B-T', 'C7B+T', 'HELIX']
MINIMA_ANGLES =  [[ -82,  175,    0],
       [  82, -176,    0],
       [ -61,  -50,  -11],
       [  63,   49,   11],
       [-135,   74,   -3],
       [ 128,  -71,    2],
       [ -82,  174,  180],
       [  83, -177, -180],
       [ -59,  -64,  176],
       [  59,   64, -177],
       [ -82,   80, -177],
       [ 105, -185,  178],
       [ -60,  -50,  180]]

MINIMA_DICT = dict(zip(MINIMA_FILENAMES, MINIMA_ANGLES))

FILENAMES = {
    'A': 'ALAp',
    'R': 'ARGp',
    'N': 'ASNp',
    'D': 'ASPd',
    'C': 'CYSd',
    'E': 'GLUd',
    'Q': 'GLNp',
    'G': 'GLYp',
    'H': 'HSDp',
    '+': 'HSEp',
    '@': 'HSPp',
    'I': 'ILEp',
    'L': 'LEUp',
    'K': 'LYSp',
    'M': 'METd',
    'F': 'PHEp',
    'P': 'PROd',
    'S': 'SERp',
    'T': 'THRp',
    'W': 'TRPp',
    'Y': 'TYRp',
    'V': 'VALp',
    '1': 'NTBp',
    '2': 'NPHp',
    '3': 'BTM',
    '4': 'BTE',
    '5': 'CTM',
    '6': 'CTE',
    '7': 'FTM',
    '8': 'FTE',
    '9': 'ITM',
    '0': 'ITE',
    'Z': 'NSPE',
    'X': 'NAEk',
    'J': 'EME',
    'O': 'NOE',
    'U': 'DECp',
    'z': 'NRPE'
}

endcap_params = {
    "nterm" : 'ACEp_f', 
    "cterm" : 'NMEp_f', 
    "n_indexname" : 'CLP', 
    "c_indexname" : 'NL'
}

NTERM_POSITION = np.zeros(3)
CTERM_POSITION = np.zeros(3)

ELEMENT_MAP = {
    # Backbone
    'NL': 'N', 'CLP': 'C', 'OL': 'O',
    # ACE cap
    'CAY': 'C', 'CLP': 'C', 'OL': 'O',
    'HY1': 'H', 'HY2': 'H', 'HY3': 'H',
    # NME/NH2 cap
    'HT1': 'H', 'HT2': 'H', 'HT3': 'H',
    'HN1': 'H', 'HN2': 'H', 'HN3': 'H',
    # Common sidechain elements by prefix
    'CLL':'Cl', 'BRL':'Br', 'SG':'S', 
}

def get_element(atom_name):
    name = atom_name.strip()
    if name in ELEMENT_MAP:
        return ELEMENT_MAP[name]
    # Fall back to first non-numeric character
    for ch in name:
        if ch.isalpha():
            # Common disambiguations
            if name.startswith('CL') or name == 'CLL':
                return 'CL'
            if name.startswith('BR'):
                return 'BR'
            if name.startswith('ZN'):
                return 'ZN'
            return ch.upper()
    return ' '

def format_element(atom_name):
    el = get_element(atom_name)
    return el.rjust(2)

def append_element(line, atom_name):
    # Strip existing newline and any trailing segment id/element/charge fields
    # (cols 67-80), pad to 76 chars, then write a single element field in cols 77-78.
    # mdtraj reads the element strictly from cols 77-78 (pdb_line[76:78]), so leaving
    # the original line's stale element there and appending a new one afterwards
    # causes mdtraj to load the wrong (old) atom type.
    line = line.rstrip('\n')
    line = line[:66].ljust(76)
    line = line + format_element(atom_name) + '\n'
    return line

def get_dihedral(traj, dihedral, i):
    dihedral_list = ["phi", "psi", "omega"]
    atom_angles = [["C", "N", "CA", "C"], ["N", "CA", "C", "N"], ["CA", "C", "N", "CA", "CAT"]]
    adjustment = [[-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1, 1]]
    k = dihedral_list.index(dihedral)
    at_angles = atom_angles[k]
    res_nums = np.array(adjustment[k]) + i
    search_strings = ["resid " + str(rn) + " and name " + str(ang) for rn, ang in zip(res_nums, at_angles)]
    search_indices = traj.topology.select(" or ".join(search_strings))
    return md.compute_dihedrals(traj, [search_indices])
    
def make_polyglycine_angles(sequence, angles, default_letter='G'):
    global NTERM_POSITION
    global CTERM_POSITION
    ##MAKE POLYGLYCINE 
    if angles.ndim == 1:
        assert len(angles) == 3
        all_angles = np.tile(angles, (len(sequence) + 2, 1))
    else:
        assert angles.ndim == 2
        assert angles.shape[0] == len(sequence) + 2
        assert angles.shape[1] == 3
        all_angles = angles
    new_seq = "G" + sequence + "G"
    for i in range(len(sequence) + 2):
        if new_seq[i] == 'P':
            geo = Geometry.geometry("P")
        else:
            geo = Geometry.geometry(default_letter)
        geo.phi = all_angles[i, 0]
        geo.psi_im1 = all_angles[i, 1]
        geo.omega = all_angles[i, 2]

        if i == 0:
            structure = PeptideBuilder.initialize_res(geo)
        else:
            PeptideBuilder.add_residue(structure, geo)
    
    out = PDBIO()
    out.set_structure(structure)
    out.save(f"template_{sequence}.pdb")
    #SET CTERM/NTERM POSITIONS

    template = md.load(f"template_{sequence}.pdb", standard_names=False)
    NTERM_POSITION = template.xyz[0, template.top.select("resSeq 1 and name C")[0]]
    CTERM_POSITION = template.xyz[0, template.top.select("resSeq {} and name N".format(len(sequence) + 2))[0]]


    #MAKE -H AND -NH STRUCTURES
    command = f"obabel template_{sequence}.pdb -O template_{sequence}_h.pdb -h"
    subprocess.run(command.split(), stdout=subprocess.PIPE)
    template_h = md.load(f'template_{sequence}_h.pdb', standard_names=False)
    top = copy.deepcopy(template_h.top)
    carbon_top = copy.deepcopy(template_h.top)
    onswitch_nitrogen = np.zeros(top.n_atoms, dtype=np.bool_)
    onswitch_carbon = np.zeros(top.n_atoms, dtype=np.bool_)

    backbone_idxs = top.select('backbone or element C')
    onswitch_nitrogen[backbone_idxs] = 1
    onswitch_carbon[backbone_idxs] = 1

    nterm_coord_exists = False
    ha_names = ['HA1', 'HA2', '', '']
    hbc_names = ['HBA', 'HBB', 'HBC']
    if default_letter == 'A':
        ha_names = ['HA', '', '']
    ha_counters = np.zeros(len(new_seq), dtype=np.int32)
    hbc_counters = np.zeros(len(new_seq), dtype=np.int32)

    pro_counters = [defaultdict(int) for k in range(len(new_seq))]
    for bond in template_h.top.bonds:
        if bond[1].element.name == 'hydrogen':
            if bond[0].element.name == 'nitrogen':
                if bond[0].residue.resSeq == 1 and not nterm_coord_exists:
                    nterm_coord_exists = True
                else:
                    onswitch_nitrogen[bond[1].index] = 1
                    if new_seq[bond[0].residue.index] == 'G':
                        onswitch_carbon[bond[1].index] = 1
                        carbon_top.atom(bond[1].index).name = 'HN'
            if bond[0].name == 'CA':
                onswitch_carbon[bond[1].index] = 1
                onswitch_nitrogen[bond[1].index] = 1
                if bond[0].residue.name == 'PRO':
                    carbon_top.atom(bond[1].index).name = 'HA'
                else:
                    carbon_top.atom(bond[1].index).name = ha_names[ha_counters[bond[0].residue.index]]
                ha_counters[bond[0].residue.index] += 1
            elif bond[0].residue.name == 'PRO' and len(bond[0].name) > 1:
                onswitch_carbon[bond[1].index] = 1
                onswitch_nitrogen[bond[1].index] = 1
                carbon_top.atom(bond[1].index).name = "H" + bond[0].name[1] + str(pro_counters[bond[0].residue.index][bond[0].name] + 1)
                pro_counters[bond[0].residue.index][bond[0].name] += 1
            elif bond[0].name == 'CB':
                onswitch_carbon[bond[1].index] = 1
                onswitch_nitrogen[bond[1].index] = 1
                carbon_top.atom(bond[1].index).name = hbc_names[hbc_counters[bond[0].residue.index]]
                hbc_counters[bond[0].residue.index] += 1
    
    for i, (ha_counter, pro_counter) in enumerate(zip(ha_counters, pro_counters)):
        if new_seq[i] == 'P':
            if ha_counter != 1:
                print(f"Proline should have 1 HA. You have {ha_counter}. OpenBabel could not place HA correctly due to aphysical bond angles. Try a different dihedral configuration.")
                _osremove(f"temp_{filename}")
                _osremove(f'template_{sequence}.pdb')
                _osremove(f'template_{sequence}_h.pdb')
                raise ValueError(f"Proline should have 1 HA. You have {ha_counter}. OpenBabel could not place HA correctly due to aphysical bond angles. Try a different dihedral configuration.")
            for key, counter in pro_counter.items():
                if counter != 2 and key != 'CA':
                    print(f"Your peptoid should have 2 {key}s. You have {counter}. OpenBabel could not place {key} correctly due to aphysical bond angles. Try a different dihedral configuration.")
                    # _osremove(f'template_{sequence}.pdb')
                    # _osremove(f'template_{sequence}_h.pdb')
                    _osremove(f"temp_{filename}")
                    raise ValueError(f"Your peptoid should have 2 {key}s. You have {counter}. OpenBabel could not place {key} correctly due to aphysical bond angles. Try a different dihedral configuration.")
        else:
            if default_letter == 'G':
                if ha_counter != 2:
                    print(f"Your amino acid {FILENAMES[new_seq[i]]} should have 2 HAs. You have {ha_counter}. OpenBabel could not place HAs correctly due to aphysical bond angles. Try a different dihedral configuration.")
                    _osremove(f"temp_{filename}")
                    _osremove(f'template_{sequence}.pdb')
                    _osremove(f'template_{sequence}_h.pdb')
                    raise ValueError(f"Your amino acid should have 2 HAs. You have {ha_counter}. OpenBabel could not place HAs correctly due to aphysical bond angles. Try a different dihedral configuration.")
            else:
                if ha_counter != 1:
                    _osremove(f"temp_{filename}")
                    _osremove(f'template_{sequence}.pdb')
                    _osremove(f'template_{sequence}_h.pdb')
                    print(f"Your amino acid {FILENAMES[new_seq[i]]} should have 1 HA. You have {ha_counter}. OpenBabel could not place HAs correctly due to aphysical bond angles. Try a different dihedral configuration.")
                    raise ValueError(f"Your amino acid should have 1 HA. You have {ha_counter}. OpenBabel could not place HAs correctly due to aphysical bond angles. Try a different dihedral configuration.")
    for i in range(top.n_atoms - 1, -1, -1):
        if not onswitch_nitrogen[i]:
            top.delete_atom_by_index(i)
    for i in range(carbon_top.n_atoms - 1, -1, -1):
        if not onswitch_carbon[i]:
            carbon_top.delete_atom_by_index(i)

    new_traj = md.Trajectory(template_h.xyz[0, onswitch_carbon], carbon_top)
    real = new_traj.atom_slice(new_traj.top.select("resSeq 2 to " + str(len(sequence) + 1)))    
    real.save_pdb(f"template_{sequence}.pdb")
    
    new_traj = md.Trajectory(template_h.xyz[0, onswitch_nitrogen], top)
    real_nh = new_traj.atom_slice(new_traj.top.select("resSeq 2 to " + str(len(sequence) + 1)))
    real_nh.save_pdb(f"template_{sequence}_nh.pdb")
    
    
    for suffix in ["", "_h", "_nh"]:
        for resid in range(2, len(sequence) + 2):
            aa_str = 'GLY'
            if default_letter == 'A':
                aa_str = 'ALA'
                commands = f'sed|-i|s/CB /CBC/g|template_{sequence}{suffix}.pdb'
                subprocess.run(commands.split('|'), stdout=subprocess.PIPE)
            if new_seq[resid - 1] == 'P':
                aa_str = 'PRO'
            if resid > 10:
                commands = f'sed|-i|s/{aa_str} A  {resid}/{aa_str} A  {resid - 1}/g|template_{sequence}{suffix}.pdb'
            elif resid == 10:
                commands = f'sed|-i|s/{aa_str} A  10/{aa_str} A   9/g|template_{sequence}{suffix}.pdb'
            else:
                commands = f'sed|-i|s/{aa_str} A   {resid}/{aa_str} A   {resid - 1}/g|template_{sequence}{suffix}.pdb'
            subprocess.run(commands.split('|'), stdout=subprocess.PIPE)


def make_peptide_angles(sequence, angles):
    global NTERM_POSITION
    global CTERM_POSITION
    if angles.ndim == 1:
        assert len(angles) == 3
        all_angles = np.tile(angles, (len(sequence) + 2, 1))
    else:
        assert angles.ndim == 2
        assert angles.shape[0] == len(sequence) + 2
        assert angles.shape[1] == 3
        all_angles = angles

    new_seq = 'G' + sequence + 'G'
    for i in range(len(new_seq)):
        geo = Geometry.geometry(new_seq[i])
        geo.phi = all_angles[i, 0]
        geo.psi_im1 = all_angles[i, 1]
        geo.omega = all_angles[i, 2]

        if i == 0:
            structure = PeptideBuilder.initialize_res(geo)
        else:
            PeptideBuilder.add_residue(structure, geo)
    
    out = PDBIO()
    out.set_structure(structure)
    out.save(f"temp_{sequence}.pdb")
    template = md.load(f"temp_{sequence}.pdb")
    NTERM_POSITION = template.xyz[0, template.top.select("resSeq 1 and name C")[0]]
    CTERM_POSITION = template.xyz[0, template.top.select("resSeq {} and name N".format(len(sequence) + 2))[0]]
    new = template.atom_slice(template.topology.select(f"resSeq 2 to {len(sequence) + 1}"))
    new.save_pdb(f"temp_{sequence}.pdb")
    for resid in range(2, len(sequence) + 2):
        if sequence[resid - 2] == 'H':
            aa_str = 'HIS'
        else:
            aa_str = FILENAMES[sequence[resid - 2]][:-1]
        if resid > 10:
            commands = f'sed|-i|s/{aa_str} A  {resid}/{aa_str} A  {resid - 1}/g|temp_{sequence}.pdb'
        elif resid == 10:
            commands = f'sed|-i|s/{aa_str} A  10/{aa_str} A   9/g|temp_{sequence}.pdb'
        else:
            commands = f'sed|-i|s/{aa_str} A   {resid}/{aa_str} A   {resid - 1}/g|temp_{sequence}.pdb'
        subprocess.run(commands.split('|'), stdout=subprocess.PIPE)
    # command = "obabel template.pdb -O template_h.pdb -h"
    # subprocess.run(command.split(), stdout=subprocess.PIPE)
    # template = md.load("template.pdb", standard_names=False)
    # template_h = md.load('template_h.pdb', standard_names=False)
    # NTERM_POSITION = template.xyz[0, template.top.select("resSeq 1 and name C")[0]]
    # CTERM_POSITION = template.xyz[0, template.top.select("resSeq {} and name N".format(len(sequence) + 2))[0]]
    # real_h = template_h.atom_slice(template_h.top.select("resSeq 2 to " + str(len(sequence) + 1)))
    # real_h.save_pdb("template.pdb")
    # os.remove("template_h.pdb")
    real_h = md.load(f"temp_{sequence}.pdb")
    
    return real_h
    
def load_sequence(sequence, filename):
    
    backbone = md.load(f"template_{sequence}.pdb", standard_names=False)
    template = md.load(f"template_{sequence}_nh.pdb", standard_names=False)
   
    nres = backbone.topology.n_residues
    i = 1
    for letter in sequence:
        # backbone_indices.append(np.arange(cur_index, cur_index + 4))
        if letter not in FILENAMES.keys():
            err_string = str(letter) + " is not a valid amino acid code."
            raise ValueError(err_string)
        
        # Read the structure PDB file
        filename2 = ".nobkb_residue_pdb/" + FILENAMES[letter] + ".pdb"
        if letter != 'P' and letter != 'G':
            structure = md.load(filename2, standard_names=False)
            #Place sidechain where the N-hydrogen used to be
            cur_h = template.topology.select("resSeq " + str(i) + " and name H")[0]
            cur_n = template.topology.select("resSeq " + str(i) + " and name N")[0]
            if letter != "A" and letter != "G":
                #Adjust orientation of side chain, pointing away from the backbone to avoid clash
                first_bond = template.xyz[0, cur_h] - template.xyz[0, cur_n]
                cur_orientation = np.mean(structure.xyz[0, 1:], axis=0) - structure.xyz[0, 0]
                rmat = rotmat(Vector(*cur_orientation), Vector(*first_bond))
                structure.xyz[0] = (np.matmul(rmat, structure.xyz[0].T)).T
            bond = template.xyz[0, cur_h] - template.xyz[0, cur_n]
            #increase C--N bond length to a normal C--N bond
            if letter == 'G':
                location = template.xyz[0, cur_n] + bond
            else:
                location = template.xyz[0, cur_n] + bond * 1.5
            translate = location - structure.xyz[0, 0]
            structure.xyz[0] += translate

            #Adjust atom indices via MDTraj's dataframe capabilities

            t, b = structure.topology.to_dataframe()
            t['resSeq'] = i
            structure = md.Trajectory(structure.xyz, md.Topology.from_dataframe(t, b))
            backbone = backbone.stack(structure)
            table, bonds = backbone.topology.to_dataframe()
            resname = list(table.loc[table['chainID'] == 1, 'resName'])[0]
            table['chainID'] = 0
            table.loc[table['resSeq'] > nres, 'resSeq'] = i
            table.loc[table['resSeq'] == i, 'resName'] = resname
            table.loc[:, 'serial'] = np.arange(len(table)) + 1
            top = md.Topology.from_dataframe(table, bonds)
            t2, b2 = top.to_dataframe()
            proper_indices = t2['serial'].to_numpy() - 1
            backbone = md.Trajectory(backbone.xyz[0, proper_indices], top)
            
        i += 1
    
    backbone.save_pdb(filename)
    

    with open(filename, "r") as f:
        lines = f.readlines()

    with open(filename, "w") as f:
        #remove unnecessary bonds from file
        for line in lines:
            if 'UNK' in line or 'CONECT' in line:
                continue
            f.write(line)
   

def _osremove(f):
    if os.path.isfile(f):
        os.remove(f)



def attach_endcap(filename, nterm, cterm, n_indexname, c_indexname):
    #Attach end caps to molecule 
    molecule = md.load(filename, standard_names=False)
    ace = md.load(f'.nobkb_residue_pdb/{nterm}.pdb', standard_names=False)
    nme = md.load(f'.nobkb_residue_pdb/{cterm}.pdb', standard_names=False)
    ace_index = ace.topology.select(f'name {n_indexname}')[0]
    ace_attach = ace.xyz[0, ace_index]
    n_translate =  NTERM_POSITION - ace_attach
    ace.xyz[0] += n_translate    
    n_bond = NTERM_POSITION - molecule.xyz[0, molecule.topology.select("resid 0 and (name N or name NT)")[0]]
    nme_attach = nme.xyz[0, nme.topology.select(f'name {c_indexname}')[0]]
    nres = len(molecule.topology.select('name CA'))
    c_translate = CTERM_POSITION - nme_attach
    nme.xyz[0] += c_translate
    c_bond = CTERM_POSITION - molecule.xyz[0, molecule.topology.select("resid " + str(nres - 1) + " and name C")[0]]
    i = nres + 1
    for cap, idx, bond in zip((ace, nme), (0, nres), (n_bond, c_bond)):
        first = np.copy(cap.xyz[0, 0])
        cap.xyz[0] -= first
        cur_orientation = np.mean(cap.xyz[0, cap.topology.select("name H or name HN1 or name HN2 or not element H")[1:]], axis=0)
        rmat = rotmat(Vector(*cur_orientation), Vector(*bond))
        cap.xyz[0] = (np.matmul(rmat, cap.xyz[0].T)).T
        cap.xyz[0] += first
        molecule = molecule.stack(cap)

        table, bonds = molecule.topology.to_dataframe()
        table.loc[table['chainID'] == 1, 'resSeq'] = i 
        i += 1
        table['chainID'] = 0
        top = md.Topology.from_dataframe(table, bonds)

        molecule = md.Trajectory(molecule.xyz, top)
    molecule.save_pdb("debug.pdb")
    molecule.save_pdb(filename)

def fix_peptide_forcefield(sequence, filename):
    with open(filename) as f:
        lines = f.readlines()

    atom_number = 1
    backbone_bonds = []
    last_c_serial = None
    ace_serials = {}
    nme_serials = {}
    nme_resnum = len(sequence) + 1
    nme_resstr = " " * (4 - len(str(nme_resnum))) + str(nme_resnum)

    def next_serial():
        nonlocal atom_number
        s = atom_number
        atom_number += 1
        return s

    def reserial(line, s):
        # Replace only the 5-char serial field (positions 6-10); position 11 space is preserved
        return line[:6] + (" " * (5 - len(str(s))) + str(s)) + line[11:]

    with open(filename, "w") as f:
        # Pass 1: MODEL and ACE (must appear first in output, before sequence residues)
        for line in lines:
            if "MODEL" in line:
                f.write(line)
            if "ACE" in line and line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                s = next_serial()
                ace_serials[atom_name] = s
                if atom_name == "C":
                    last_c_serial = s
                # Replace serial and set resSeq to 0 (positions 22-25)
                l = reserial(line, s)
                l = l[:22] + "   0" + l[26:]
                f.write(l)

        # Pass 2: sequence residues, NME/NH2, END
        # Skip all MDTraj CONECT records — they use global atom indices, not PDB serials
        for line in lines:
            if "ENDMDL" in line or line.strip() == "END":
                # Inter-residue backbone bonds
                for c_ser, n_ser in backbone_bonds:
                    f.write(f'CONECT{c_ser:5d}{n_ser:5d}\n')
                    f.write(f'CONECT{n_ser:5d}{c_ser:5d}\n')
                # ACE internal bonds: C-O, C-CAY, CAY-HY1/2/3
                if ace_serials:
                    c, o, cay = ace_serials.get('C'), ace_serials.get('O'), ace_serials.get('CAY')
                    for a, b in [(c, o), (c, cay)]:
                        if a and b:
                            f.write(f'CONECT{a:5d}{b:5d}\n')
                            f.write(f'CONECT{b:5d}{a:5d}\n')
                    for h in ['HY1', 'HY2', 'HY3']:
                        if cay and h in ace_serials:
                            f.write(f'CONECT{cay:5d}{ace_serials[h]:5d}\n')
                            f.write(f'CONECT{ace_serials[h]:5d}{cay:5d}\n')
                # NME internal bonds: N-H, N-CAT, CAT-HT1/2/3
                if nme_serials:
                    n, cat = nme_serials.get('N'), nme_serials.get('CAT')
                    for a, b in [(n, nme_serials.get('H')), (n, cat)]:
                        if a and b:
                            f.write(f'CONECT{a:5d}{b:5d}\n')
                            f.write(f'CONECT{b:5d}{a:5d}\n')
                    for h in ['HT1', 'HT2', 'HT3']:
                        if cat and h in nme_serials:
                            f.write(f'CONECT{cat:5d}{nme_serials[h]:5d}\n')
                            f.write(f'CONECT{nme_serials[h]:5d}{cat:5d}\n')
                f.write(line)
                continue
            if "ACE" in line or 'MODEL' in line or line.startswith('CONECT') or 'CRYST' in line or 'REMARK' in line:
                continue
            elif "NME" in line or line[17:20] == 'NH2':
                orig_name = line[12:16].strip()
                s = next_serial()
                # Rename atom name first on original line, then reserial, then fix resSeq
                l = line
                if orig_name == "NT":
                    if last_c_serial is not None:
                        backbone_bonds.append((last_c_serial, s))
                        last_c_serial = None
                    nme_serials['N'] = s
                    l = l[:12] + " N  " + l[16:]
                elif orig_name == "HNT":
                    nme_serials['H'] = s
                    l = l[:12] + " H  " + l[16:]
                else:
                    nme_serials[orig_name] = s
                l = reserial(l, s)
                l = l[:22] + nme_resstr + l[26:]
                if "TER" in line:
                    l = l.rstrip('\n') + '\n'
                f.write(l)
            elif line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                s = next_serial()
                if atom_name == "N":
                    if last_c_serial is not None:
                        backbone_bonds.append((last_c_serial, s))
                    last_c_serial = None
                elif atom_name == "C":
                    last_c_serial = s
                f.write(reserial(line, s))
            else:
                f.write(line)
def pdb_to_peptoid_forcefield(sequence, filename, cter):
    with open(filename) as f:
        lines = f.readlines()
    atom_number = 1
    cterm_counter = 1
    backbone_bonds = []  # (c_serial, n_serial) for inter-residue backbone bonds
    last_c_serial = None
    with open(filename, "w") as f:
        #Renumber atoms and apply standard OpenMM/CHARMM atom names for backbone and endcaps.
        for line in lines:
            if "MODEL" in line:
                f.write(line)
            if "ACE" in line:
                numlen = len(str(atom_number))
                str0 = " " * (5 - numlen) + str(atom_number) + "  "
                atom_number += 1
                str2 = "p    " + "0   "
                if line[13:16] == "CH3":
                    str1 = "CAY"
                elif line[13:18] == "H1  A":
                    str1 = "HY1"
                elif line[13:18] == "H2  A":
                    str1 = "HY2"
                elif line[13:18] == "H3  A":
                    str1 = "HY3"
                else:
                    str1 = line[13:16]
                if str1.strip() == "CLP":
                    last_c_serial = atom_number - 1
                l = line[:6] + str0 + str1 + line[16:20] + str2 + line[29:]
                l = append_element(l, str1)
                f.write(l)

        for line in lines:
            l = line
            if "ENDMDL" in line:
                for c_ser, n_ser in backbone_bonds:
                    f.write(f'CONECT{c_ser:5d}{n_ser:5d}\n')
                    f.write(f'CONECT{n_ser:5d}{c_ser:5d}\n')
                f.write(line)
                continue
            if "ACE" in line or 'MODEL' in line or 'CONECT' in line or 'CRYST' in line or 'REMARK' in line:
                continue
            elif "NME" in line or cter in line[17:22]:
                numlen = len(str(atom_number))
                str0 = " " * (5 - numlen) + str(atom_number) + "  "
                atom_number += 1
                str2 = "p   "
                nres = len(sequence)
                if nres > 9:
                    str3 = str(nres + 1) + "   "
                else:
                    str3 = " " + str(nres + 1) + "  "
                if line[13:15] == "H ":
                    if cter == 'NH2':
                        str1 = "HN" + str(cterm_counter)
                        cterm_counter += 1
                    else:
                        str1 = line[13:16]
                elif line[13:18] == "H1  N":
                    str1 = "HT1"
                elif line[13:18] == "H2  N":
                    str1 = "HT2"
                elif line[13:18] == "H3  N":
                    str1 = "HT3"
                else:
                    str1 = line[13:16]
                if str1.strip() == "NL" and last_c_serial is not None:
                    backbone_bonds.append((last_c_serial, atom_number - 1))
                if "TER" in line:
                    l = line[:6] + str0 + str1 + line[16:20] + str2 + str3 + "\n"
                else:
                    l = line[:6] + str0 + str1 + line[16:20] + str2 + str3 + line[29:]
                    l = append_element(l, str1)

            elif line.startswith("ATOM"):
                num_str = line[24:26]
                cur_resnum = int(num_str)
                cur_residue = " " + FILENAMES[sequence[cur_resnum - 1]] + "   "
                numlen = len(str(atom_number))
                str0 = " " * (5 - numlen) + str(atom_number) + " "
                atom_number += 1

                if line[13:18] == "HA3 G":
                    str1 = " HA1"
                # elif line[12:16] == " HN ":
                #     str1 = " H  "
                elif line[12:16] == " N  ":
                    str1 = " NL "
                elif line[12:16] == " C  ":
                    str1 = " CLP"
                elif line[12:16] == " O  ":
                    str1 = " OL "
                else:
                    str1 = line[12:16]
                name = str1.strip()
                if name == "NL":
                    if last_c_serial is not None:
                        backbone_bonds.append((last_c_serial, atom_number - 1))
                    last_c_serial = None
                elif name == "CLP":
                    last_c_serial = atom_number - 1
                l = line[:6] + str0 + str1 + cur_residue + line[24:]
                l = append_element(l, str1)
            f.write(l)




def create_peptoid(sequence, angle_seq, filename="molecule.pdb", default_letter='G', endcap_par=endcap_params):
    if not filename.endswith('.pdb'):
        filename = filename + '.pdb'
    make_polyglycine_angles(sequence, angle_seq, default_letter=default_letter)
    load_sequence(sequence, f"temp_{filename}")
    attach_endcap(f"temp_{filename}", **endcap_par)
    if default_letter == 'G':
        pdb_to_peptoid_forcefield(sequence, f"temp_{filename}", endcap_params['cterm'].replace("p_f", ""))
    os.rename(f"temp_{filename}", filename)
    _osremove(f'template_{sequence}.pdb')
    _osremove(f'template_{sequence}_h.pdb')
    _osremove(f'template_{sequence}_nh.pdb')

    return md.load(filename, standard_names=False)

def create_peptide(sequence, angle_seq, filename="peptide.pdb", default_letter='G', endcap_par=endcap_params):
    make_peptide_angles(sequence, angle_seq)
    attach_endcap(f"temp_{sequence}.pdb", **endcap_par)
    fix_peptide_forcefield(sequence, f"temp_{sequence}.pdb")
    os.rename(f"temp_{sequence}.pdb", filename)

def _run_tide_or_toid(tide_or_toid):
    if tide_or_toid:
        return create_peptide
    else:
        return create_peptoid

def run():
    parser = argparse.ArgumentParser(
        description="Molecule Inserter for peptoids"
    )
    parser.add_argument(
        "--seq",
        type=str,
        default=None,
        metavar="seq",
        help="Input Sequence"
    )
    parser.add_argument(
        "--mini",
        type=str,
        default=None,
        metavar="mini",
        help="minimum configuration"
    )
    parser.add_argument(
        "--phi",
        type=int,
        default=None,
        metavar="phi",
        help="minimum configuration"
    )
    parser.add_argument(
        "--psi",
        type=int,
        default=None,
        metavar="psi",
        help="minimum configuration"
    )
    parser.add_argument(
        "--omega",
        type=int,
        default=None,
        metavar="omega",
        help="minimum configuration"
    )
    parser.add_argument(
        "--anglefile",
        type=str,
        default=None,
        metavar="anglefile",
        help="minimum configuration"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="molecule",
        metavar="file",
        help="filename under which to save final structure"
    )
    parser.add_argument(
        "--tide",
        action="store_true",
        help="Make a peptide, not a peptoid"
    )
    parser.add_argument(
        "--nobackup",
        action="store_true",
        help="Overwrite existing files."
    )
    parser.add_argument(
        "--cter",
        type=str,
        default="nme",
        metavar="cter",
        help="C-Terminus"
    )
    parser.add_argument(
        "--nsa",
        action="store_true",
        help="N-substituted alanine"
    )
    args = parser.parse_args()

    if args.seq is None:
        raise ValueError("Sequence Entry Needed")

    # Load the minimum that the peptoid is in
    default_letter = 'G'
    if args.nsa:
        default_letter = 'A'
    if args.file.endswith(".pdb"):
        fname = args.file
    else:
        fname = args.file + ".pdb"
    path = os.path.join(os.getcwd(), fname)
    num_backups = 0
    if not args.nobackup:
        while os.path.exists(path):
            if num_backups == 0:
                print("WARNING: " + fname[:-4]  + ".pdb already exists. Backing up file to " + fname[:-4] + "1.pdb")
            else:
                print("WARNING: " + fname[:-4] + str(num_backups) + ".pdb already exists. Backing up file to " + fname[:-4] + str(num_backups + 1) + ".pdb")
            num_backups += 1
            path = os.path.join(os.getcwd(), fname[:-4] + str(num_backups) + ".pdb")
        while num_backups > 0:
            if num_backups == 1:
                  command = "cp " + fname + " " + fname[:-4] + "1.pdb"
            else:
                  command = "cp " + fname[:-4] + str(num_backups - 1) + ".pdb" + " " + fname[:-4] + str(num_backups) + ".pdb"
            subprocess.run(command.split(), stdout=subprocess.PIPE)
            num_backups -= 1

    # assert args.mini is not None or (args.phi is not None and args.psi is not None and args.omega is not None) or args.anglefile is not None

    if args.cter == 'NH2' and not args.tide:
        endcap_params['cterm'] = 'NH2p_f'
    elif args.cter == 'NDM':
        endcap_params['cterm'] = 'NDMp_f'
    elif args.tide:
        if args.cter == 'NH2':
            endcap_params['cterm'] = 'NH2_f'
        else:
            endcap_params['cterm'] = 'NME_f'
        endcap_params['c_indexname'] = 'NT'
        endcap_params['nterm'] = 'ACE_f'
        endcap_params['n_indexname'] = 'C'
        
    if args.mini is not None:
        try:
            angles = np.array(MINIMA_DICT[args.mini])
        except:
            raise ValueError("Ramachandran minimum error -- minimum mist be one of: " + str(MINIMA_FILENAMES) + ". Your minimum is " + args.mini)
        _run_tide_or_toid(args.tide)(args.seq, angles, fname, default_letter)
        return
    if args.anglefile is not None:
        try:
            base_path=os.path.abspath(os.path.join(os.getcwd(), *(['..'] * (len(os.path.abspath(os.getcwd()).split(os.sep)) - os.path.abspath(os.getcwd()).split(os.sep).index('toid_explore') - 1)))) 
            filename=f"{base_path}/{args.anglefile}"

            angles = np.loadtxt(filename)
        except:
            raise ValueError(f"Angle file error -- Could not load text file as numpy: {filename}")
        if np.all(angles < 2 * np.pi) and np.all(angles > -np.pi):
            angles *= (180 / np.pi)
        _run_tide_or_toid(args.tide)(args.seq, angles, fname, default_letter)
        return
    if args.phi is not None and args.psi is not None and args.omega is not None:
        angles = np.array([args.phi, args.psi, args.omega])
        _run_tide_or_toid(args.tide)(args.seq, angles, fname, default_letter)
        return
    else:
        print("Input Error -- Must enter: minimum (--mini) OR phi/psi/omega triplet (--phi, --psi, --omega) OR text file containing angles (--anglefile)")
        return
run()


            
