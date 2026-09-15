# Imports
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import os
import argparse
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent

def add_bonds_from_forcefield(topology, forcefield):
    """Add missing bonds to a topology using force field templates."""
    
    existing = set()
    for bond in topology.bonds():
        existing.add((bond[0].index, bond[1].index))
        existing.add((bond[1].index, bond[0].index))
    
    def safe_add(a1, a2):
        if (a1.index, a2.index) not in existing:
            topology.addBond(a1, a2)
            existing.add((a1.index, a2.index))
            existing.add((a2.index, a1.index))
    
    for res in topology.residues():
        if res.name not in forcefield._templates:
            continue
        template = forcefield._templates[res.name]
        atom_map = {a.name: a for a in res.atoms()}
        template_atoms = list(template.atoms)
        n_atoms = len(template_atoms)
        
        # Skip bonds involving external atom indices
        external_indices = set(template.externalBonds)
        
        for bond in template.bonds:
            # Skip if either index is out of range
            if bond[0] >= n_atoms or bond[1] >= n_atoms:
                continue
            # Skip if an bond is an external bond (i.e., connects to another residue)
            # if bond[0] in external_indices or bond[1] in external_indices:
            #     continue
            a1_name = template_atoms[bond[0]].name
            a2_name = template_atoms[bond[1]].name
            if a1_name in atom_map and a2_name in atom_map:
                safe_add(atom_map[a1_name], atom_map[a2_name])

def run_simulation():
    parser = argparse.ArgumentParser(
        description="Run a brief simulation for peptoid in OpenMM"
    )
    parser.add_argument(
        "--pdb",
        type=str,
        required=True,
        metavar="pdb",
        help="Path to the PDB file of the peptoid"
    )

    # Find structure (pdb) and force field (xml) files
    args = parser.parse_args()
    pdb = PDBFile(args.pdb)
    Topology.loadBondDefinitions(f"{REPO_ROOT}/spce_mosic.xml")
    forcefield = ForceField(f"{REPO_ROOT}/spce_mosic.xml")
    add_bonds_from_forcefield(pdb.topology, forcefield)
    modeller = Modeller(pdb.topology, pdb.positions)
    residue_map = {res: res.name for res in modeller.topology.residues()}
    modeller.addSolvent(forcefield, 
                        boxSize=Vec3(5.0, 5.0, 5.0)*nanometers,
                        model='spce',
                        residueTemplates=residue_map)
    residue_map = {res: res.name for res in modeller.topology.residues() if res.name not in ["HOH","CL"]} # let it figure out water/ions
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME,
            nonbondedCutoff=1*nanometer, constraints=HBonds,
            residueTemplates=residue_map)
    
    # Setup directories and move to the run directory
    sim_dir = Path(f"{REPO_ROOT}/simulations")
    sim_dir.mkdir(exist_ok=True)
    run_dir = sim_dir / Path(args.pdb).stem
    run_dir.mkdir(exist_ok=False)
    # Change to the run directory
    os.chdir(run_dir)
    
    # Run simulation
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)

    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    # Write the trajectory to a file called "output.pdb"
    simulation.reporters.append(PDBReporter('output.pdb', 100))

    # Report infomation to the screen as the simulation runs
    simulation.reporters.append(StateDataReporter(stdout, 100, step=True,
            potentialEnergy=True, temperature=True))
    
    #Run the simulation for 1000 timsteps
    print("Running simulation...")
    simulation.step(1000)
    return
run_simulation()
