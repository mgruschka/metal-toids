# GROMACS side
gmx_mpi editconf -f conf.gro -o conf_box.gro -c -d 1.2 -bt cubic
gmx_mpi grompp -f em.mdp -c conf_box.gro -p topol.top -o em.tpr
gmx_mpi mdrun -s em.tpr -rerun conf_box.gro
gmx_mpi energy -f ener.edr  # select CMAP term
