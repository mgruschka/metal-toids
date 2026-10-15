"""
Generic lambda-window machinery for TI drivers whose potential is LINEAR in
lambda, U(lambda) = U(0) + lambda * dU/dlambda (true for the alpha-TI in
TI_param/run_TI.py; the ion swap in ti.py has its own copy of the loop).

    schedules      uniform(n), gauss_legendre(n)
    run_window     NVT production at one lambda -> CSV of dU/dl (+ extra columns)
    cumulative_ti  trapezoid dG(0 -> lambda_k) at every node of a uniform ladder
    mbar_over_lambda
                   MBAR across the windows.  With U linear in lambda the
                   reduced potential of sample n in state k is
                   beta * lambda_k * dU/dl(x_n) + const(x_n), so the dU/dl
                   column is all MBAR needs -- no re-evaluation.
"""
import numpy as np
from openmm import app

from .build import log


# ---------------------------------------------------------------- schedules

def uniform(n):
    """n evenly spaced nodes on [0, 1], endpoints included."""
    if n < 2:
        raise ValueError("uniform schedule needs n >= 2")
    return np.linspace(0.0, 1.0, n)


def gauss_legendre(n):
    """nodes and weights on [0, 1] (same as ti.gauss_legendre)."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def block_sem(x, nblocks=5):
    """standard error of the mean from `nblocks` block averages (as ti.block_sem)."""
    x = np.asarray(x)
    n = len(x) // nblocks * nblocks
    if n < nblocks:
        return float("nan")
    means = x[:n].reshape(nblocks, -1).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(nblocks))


# ---------------------------------------------------------------- sampling

def run_window(sim, csv_path, lam_name, lam, n_steps, every, dt_fs, dudl_fn,
               extra=(), traj_path=None, traj_every=0):
    """
    Set `lam_name` = lam and run n_steps, writing time_ps, dudl_kJmol and one
    column per (header, fn(sim)) in `extra` every `every` steps.  Reporters
    are cleared in `finally` so a DCD is closed before the next window.
    Returns the dU/dl samples (kJ/mol).
    """
    sim.context.setParameter(lam_name, float(lam))
    if traj_path and traj_every:
        # unwrapped, as in ti.py: the complex stays whole
        sim.reporters.append(app.DCDReporter(traj_path, traj_every, enforcePeriodicBox=False))
    out = []
    try:
        with open(csv_path, "w") as fh:
            fh.write("# time_ps,dudl_kJmol" + "".join("," + h for h, _ in extra) + "\n")
            t = 0.0
            for _ in range(n_steps // every):
                sim.step(every)
                t += every * dt_fs * 1e-3
                d = dudl_fn(sim.context)
                out.append(d)
                fh.write(f"{t:.3f},{d:.6f}" + "".join(f",{fn(sim):.6f}" for _, fn in extra) + "\n")
                fh.flush()
    finally:
        sim.reporters.clear()
    return np.array(out)


def load_window(csv_path, discard_frac=0.0):
    """(dudl_kJmol, other columns) after dropping the first discard_frac."""
    a = np.atleast_2d(np.loadtxt(csv_path, delimiter=","))
    a = a[int(discard_frac * len(a)):]
    return a[:, 1], a[:, 2:]


# ---------------------------------------------------------------- estimators

def cumulative_ti(lams, means, sems):
    """
    Trapezoid dG(0 -> lams[k]) for every k, same units as `means`.  Errors
    assume independent windows: var_k = sum_i (w_ik * sem_i)^2 with the
    trapezoid weights w_ik of the partial integral.
    """
    lams, means, sems = map(np.asarray, (lams, means, sems))
    dG, err = np.zeros(len(lams)), np.zeros(len(lams))
    for k in range(1, len(lams)):
        w = np.zeros(len(lams))
        h = np.diff(lams[:k + 1])
        w[:k] += 0.5 * h
        w[1:k + 1] += 0.5 * h
        dG[k] = w @ means
        err[k] = np.sqrt(((w * sems) ** 2).sum())
    return dG, err


def mbar_over_lambda(lams, dudl_samples, kT, return_matrix=False):
    """
    MBAR free energies f_k - f_0 (same units as kT, i.e. kJ/mol in, kJ/mol
    out) for a potential linear in lambda.  Each window's dU/dl series is
    subsampled to its statistical inefficiency first (as mbar.decorrelate).
    Returns (dG, dG_err, g_per_window, adjacent_overlap), plus the full
    K x K MBAR overlap matrix as a fifth element if return_matrix.
    """
    from pymbar import MBAR, timeseries
    lams = np.asarray(lams, float)
    sub, g = [], []
    for x in dudl_samples:
        x = np.asarray(x)
        gi = timeseries.statistical_inefficiency(x)
        sub.append(x[timeseries.subsample_correlated_data(x, g=gi)])
        g.append(gi)
    D = np.concatenate(sub)
    N_k = np.array([len(s) for s in sub])
    u_kn = lams[:, None] * D[None, :] / kT
    m = MBAR(u_kn, N_k, solver_protocol="robust")
    r = m.compute_free_energy_differences()
    O = m.compute_overlap()["matrix"]
    adj = np.array([O[k, k + 1] for k in range(len(lams) - 1)])
    log(f"MBAR over lambda: N_eff per window {N_k.tolist()}, min adjacent overlap {adj.min():.3f}")
    out = (r["Delta_f"][0] * kT, r["dDelta_f"][0] * kT, np.array(g), adj)
    return out + (O,) if return_matrix else out
