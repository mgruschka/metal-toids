"""
Modified from Pfaendtner Group / Sarah Alamdari: https://github.com/UWPRG/peptoid-ffgen/blob/master/peptoid-ffgen/gromacs_utils.py
"""
import subprocess
import contextlib


def grompp(mpirun, mdp, coord, tpr, use_index=False, index="index.ndx", top="topol.top",
           maxwarn=1, np=1):
    """
    Python wrapper for gmx grompp

    Parameters
    ----------
    mpirun : Bool
        Is this a multi-node run or not (gmx vs gmx_mpi), if True must specify
        number of processes (np)
    mdp : str
        Filename of .mdp file
    coord : str
        Filename of .gro or .pdb file
    tpr : str
        Filename of output of grompp
    index : str
        Filename of index file (Default index.ndx)
    top : str
        Filename of topology file (Default topol.top)
    maxwarn : int
        Maximum number of acceptable warnings when grompping
    mpirun : bool
        Is this a multi-node run or not gmx (False) vs gmx_mpi (Default: True)
        number of processes (np)
    np : int
        Number of processes to run mpirun on (Default 1 for non-mpi run)
    """
    if mpirun == True:
        mpi = "gmx_mpi"
    elif mpirun == False:
        mpi = "gmx"
    else:
        print ("mpirun only takes bool as input")

    if use_index:
        commands = [mpi, "grompp", "-f", mdp, "-p",
                top, "-c", coord, "-r", coord, "-o", tpr, "-n", index,
                "-maxwarn", str(maxwarn)]
    else:
        commands = [mpi, "grompp", "-f", mdp, "-p",
                top, "-c", coord, "-r", coord, "-o", tpr,
                "-maxwarn", str(maxwarn)]
        
    subprocess.run(commands)
    return


def mdrun(mpirun, deffnm, plumed=False, plumed_file='plumed.dat', np=1):
    """
    Python wrapper for gmx mdrun -deffnm

    Parameters
    ----------
    deffnm : str
         File names for md run
    mpirun : bool
        Is this a multi-node run or not gmx (False) vs gmx_mpi (Default: True)
        number of processes (np)
    plumed : bool
        Turns plumed on/off
    np : int
        Number of processes to run mpirun on (Default 1 for non-mpi run)
    """
    
    if mpirun == True:
        mpi = "gmx_mpi"
    elif mpirun == False:
        mpi = "gmx"
    else:
        print ("mpirun only takes bool as input")
    
    if plumed:
        commands = [mpi, "mdrun", "-deffnm", deffnm, "-ntomp", str(np), "-v", "-plumed", plumed_file]
    else:
        commands = [mpi, "mdrun", "-deffnm", deffnm, "-ntomp", str(np), "-v"]
    
    subprocess.run(commands)
    return

def make_ndx(mpirun, scan_atoms, index_input, np=1):
    """
    Python wrapper for gmx make_ndx

    Parameters
    ----------
    index_input : str
        file (with extension) used for gmx make_ndx
    mpirun : bool
        Is this a multi-node run or not gmx (False) vs gmx_mpi (Default: True)
        number of processes (np)
    scan_coordinates : array of ints
        Indicates atoms involved in scan. Need this for running MD
    np : int
        Number of processes to run mpirun on (Default 1 for non-mpi run)
    """

    if mpirun == True:
        mpi = "mpirun " + "np " + str(np) + " gmx_mpi"
    elif mpirun == False:
        mpi = "gmx"
    else:
        print ("mpirun only takes bool as input")

    commands = [mpi, "make_ndx", "-f", index_input]
    pipe_command=["cat","cat_make_ndx.txt"] # pick ff in working directory and TIP3P ** this should not be hardcoded lmao FIX **
    ps = subprocess.Popen(pipe_command, stdout=subprocess.PIPE)
    output = subprocess.check_output(commands, stdin=ps.stdout)
    subprocess.run(commands)

    # append scan atoms to end of file
    if len(scan_atoms) != 4:
        raise Exception("Need 4 atoms to describe a dihedral")

    w1 = "[ SCAN ]\n"
    w2 = f'\t{scan_atoms[0]}\t{scan_atoms[1]}\t{scan_atoms[2]}\t{scan_atoms[3]}\n'
    f1 = open("index.ndx", "a")  # append mode
    f1.write(w1+w2)
    f1.close()

    return


def pdb2gmx(mpirun, input_pdb, output_gro, top="topol.top", ff="charmm36-feb2021", np=1, water="none", tide=False):
    """
    Python wrapper for gmx pdb2gmx

    Parameters
    ----------
    """
    
    if mpirun == True:
        mpi = "gmx_mpi"
    elif mpirun == False:
        mpi = "gmx"
    else:
        print ("mpirun only takes bool as input")

    commands = [mpi, "pdb2gmx", "-f", input_pdb, "-o", output_gro, "-p", top, "-ff", ff, "-water", water]
    # if tide:
    #     first_commands = ['echo', '3', '|', 'echo', '4', '|']
    #     last_commands = ['-ter']
    #     commands = first_commands + commands + last_commands
    if tide:
        commands.append('-ter')
        process = subprocess.Popen(commands, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

        process.communicate('3\n4\n')

    else:
        subprocess.run(commands) # selects FF in current directory
    return

def editconf(mpirun, input_gro, output_gro, use_distance=False, distance=1.3, np=1, center=False):
    """
    Python wrapper for gmx editconf

    Parameters
    ----------
    distance : int
        distance between solute and box (default 1.3)
    """
    
    if mpirun == True:
        mpi = "gmx_mpi"
    elif mpirun == False:
        mpi = "gmx"
    else:
        print ("mpirun only takes bool as input")
    if use_distance: 
        commands = [mpi, "editconf", "-f", input_gro, "-o", output_gro, '-c', '-d', str(distance)]
    else: 
        commands = [mpi, "editconf", "-f", input_gro, "-o", output_gro]
    if center:
        commands.append("-c")
    subprocess.run(commands) # selects FF in current directory

    return

# def minimize(mpirun, em="em.tpr", np=1):
#     """
#     Perform energy minimization on a peptoid
#     """
#     if mpirun == True:
#         mpi = "/project2/andrewferguson/berlaga/bin/gmx_mpi"
#     elif mpirun == False:
#         mpi = "gmx"
#     else:
#         print ("mpirun only takes bool as input")

#     commands = [mpi, "mdrun", "-ntomp", np, "-v", "-deffnm", em]
#     subprocess.run(commands)

#     return
    

def insert_molecules(mpirun, pdb, box, xdim, ydim, zdim, nmol=1):
    """
    Python wrapper for gmx insert-molecules
    Parameters 
    -----------------
    mpirun: boolean if you're using multiple processes
    pdb: molecule to be inserted
    box: box to insert into
    xdim, ydim, zdim: dimensions
    nmol: number of molecules to be inserted
    """

    if mpirun == True:
        mpi = "gmx_mpi"
    elif mpirun == False:
        mpi = "gmx"
    commands = [mpi, "insert-molecules", "-ci", pdb, "-o", box, "-nmol", str(nmol), "-box", str(xdim), str(ydim), str(zdim)]
    subprocess.run(commands)
    return 
    
def insert_molecules_cubic(mpirun, pdb, box, dim=3, nmol=1):
    insert_molecules(mpirun, pdb, box, dim, dim, dim, nmol)
    return
    
def energy(mpirun, input_f, output_xvg, energy_num=10, np=1):
    """
    Python wrapper for gmx energy

    Parameters
    ----------
    """
    if mpirun == True:
        mpi = "gmx_mpi"
    elif mpirun == False:
        mpi = "gmx"
    else:
        print ("mpirun only takes bool as input")

    commands = [mpi, "energy", "-f", input_f, "-o", output_xvg]
    subprocess.run(commands, input=str(energy_num).encode())
    return


