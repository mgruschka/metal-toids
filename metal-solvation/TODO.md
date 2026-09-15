# TODO List

This document will house a to-do list for the steps required for the simulation of metal solvation energies for the computation of the base $C_{4}$ parameters
The general procedure will come from the 20215 Li et al. paper ("Parameterization of Highly Charged Metal Ions using the 12-6-4 LJ-Type Nonbonded Model in Explicit Water"). 

## Hydration Free Energy
Following their work, the primary goa is to use thermodynamic integration (TI) to calculate the hydration free energy, using the cycle:

Dummy atom in Vacuum -> Dummy atom in water <==>  Neutral atom in water <==> Ion in water <- ion in vacuum -> Dummy in vacuum

Where <==> represents an equilibrium (see figure 1. in the original paper)

To get $\Delta G_{Hydr}$ (the HFE) we need to find $\Delta G$ for 4 separate processes:
1. $\pm \Delta G_{VDW}$
2. $\pm \Delta G_{Ele + Pol}$

Then we can calculate the HFE with: $$\Delta G_{Hydr} = 1/2 \times ( \Delta G_{VDW} + \Delta G_{Ele+Pol} - (-\Delta G_{Ele+Pol} - \Delta G_{VDW}))$$

To do this, the following steps were taken:
1. dummy atom in 32 x 32 x 32 Angstrom box, closest water molecule 1.5 Angstrom away, with a total of 722 water molecules
2. 1000 step steepest descent minim, then 1000 step conj. grad. min.
3. 500 ps heating, from 0-300K, NVT
4. Equil for 500 ps at 300 K, 1 atm NPT
    - Final snapshot = inintial structure for $\Delta G_{VDW}$
5. 4 window, linear soft-core scaling for $\Delta G_{VDW}$ ($V(\lambda) = (1-\lambda) V_0 + \lambda V_{1}$) with $\lambda = [0.1127,0.5,0.88729,0.98]$
6. Final snapshot from 5. used for starting $\Delta G_{Ele+Pol}$
7. 9 window, linear with $\lambda=[0, 0.02544, 0.12923, 0.29707, 0.5, 0.70292, 0.87076, 0.97455, 1]
8. Reverse of 7 to calculate $-\Delta G_{Ele+Pol}$ 
NOTE: For these window calculations, the first and last windows are not used to compute the free energy, rather as equilibration windows.
9. $-\Delta G_{VDW}$ is calculated with a 3-window linear with $\lambda=[0.1127,0.5,0.88729]$. 
For $\pm \Delta G_{VDW}$:
- NPT Ensemble
- 300 ps per window, (100 equil, 200 data collection)

For $\pm \Delta G_{Ele+Pol}$:
- NPT Ensemble
- 200 ps per window, last 150 ps for data collection

Uncertainty calcualted with: 
1. std of 2-block calculations of each $\Delta G$
2. Using autocorrelation times with $\tau_{A}=500 fs$ for VDW and $250 fs$ for Ele + Pol
- Should expect a range of 0.1-7.5 kcal/mol 

## Ion-Oxygen Distance and CN 
