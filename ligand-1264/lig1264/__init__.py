"""
lig1264 - 12-6-4 ion/ligand parametrization runs in OpenMM.

Pipeline (run.py):  build -> minimize (3 stages) -> heat NVT -> NPT equil/prod
Everything that varies between systems lives in a YAML config (config.py).
"""
from .config import Config, load_config  # noqa: F401
