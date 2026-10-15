#!/bin/bash
# 10-minute check (andrewferguson-gpu, the quick-turnaround partition): does the CUDA platform load with the
# env's nvrtc pin on this node's driver, and how fast is a TI window?
#SBATCH --job-name=bench1264
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=andrewferguson-gpu
#SBATCH --account=pi-andrewferguson
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/scratch/midway3/gruschka/metal-toid/bin/python
nvidia-smi -L
$PY - <<'PYEOF'
import time, os, openmm as mm
from openmm import app, unit
print("platforms:", [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())])
import sys; sys.path.insert(0, ".")
from lig1264.config import load_config
from lig1264 import build, forces
cfg = load_config("configs/ni_co_imidazole4.yaml")
out = "runs/bench"; os.makedirs(out, exist_ok=True)
csv = "../12-6-4_base_params.csv"
pA = forces.ion_params("Ni", 2, csv); pB = forces.ion_params("Co", 2, csv)
top, pos, system, groups = build.build_system_files(cfg, out, lambda s, g: forces.make_ion_alchemical(s, g["ion"][0], pA, pB))
integ = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 1.0*unit.femtosecond)
plat = mm.Platform.getPlatformByName("CUDA")
sim = app.Simulation(top, system, integ, plat, {"Precision": "mixed"})
print("CUDA device:", plat.getPropertyValue(sim.context, "DeviceName"))
sim.context.setPositions(pos); sim.context.setParameter("lambda_ion", 0.5)
sim.minimizeEnergy(maxIterations=500)
sim.context.setVelocitiesToTemperature(300*unit.kelvin)
sim.step(2000)
t0 = time.time(); n = 20000
for _ in range(n // 500):
    sim.step(500); forces.dudl(sim.context)
dt = time.time() - t0
print(f"{n} steps at 1 fs in {dt:.1f} s -> {n*1e-6/dt*86400:.1f} ns/day (with dU/dl every 0.5 ps)")
print(f"one 5 ns window ~ {5/(n*1e-6/dt*86400)*24:.1f} h; full 12-window TI + 2.4 ns prep ~ {62.4/(n*1e-6/dt*86400)*24:.1f} h")
PYEOF
