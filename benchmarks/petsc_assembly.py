"""Compare PETSc row/COO assembly in a reproducible, single-rank RIXS run.

Run from the repository root with PYTHONPATH=.; see benchmarks/README.md.
Requires PETSc, Numba, and psutil. Timings exclude setup/JIT warmup and include
matrix construction. Process peak RSS includes setup and native allocations.
"""
import argparse
from collections import Counter
from functools import partial
import gc
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

import numpy as np
import psutil
from petsc4py import PETSc

import edrixs
from edrixs.fock_basis import build_fock_basis

parser = argparse.ArgumentParser()
parser.add_argument('policy', choices=['retain', 'recompute'])
parser.add_argument('--nval', type=int, default=18)
parser.add_argument('--out', required=True)
parser.add_argument('--assembly-method', choices=['row', 'coo'], default='row')
parser.add_argument('--chunk-cols', type=int, default=1024)
parser.add_argument('--preallocation', choices=['tailored', 'default'], default='tailored')
args = parser.parse_args()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)

# Warm the same JIT assembly path on a tiny problem before measuring RIXS.
small_basis = edrixs.FockBasisSpec.from_args(4, 2, 2, 2)
warm = edrixs.build_op(
    np.eye(6, dtype=complex), None, small_basis, backend='petsc',
    backend_kws={'nnz_guess_per_row': 8, 'assembly_method': args.assembly_method},
)()
warm.destroy()

nv = args.nval
ne = nv // 2
rng = np.random.default_rng(84135)
raw = rng.normal(size=(nv, nv)) + 1j * rng.normal(size=(nv, nv))
one_body = (raw + raw.conj().T) / (8 * np.sqrt(nv))
emat_i = np.zeros((nv + 2, nv + 2), dtype=complex)
emat_i[:nv, :nv] = one_body
emat_i[nv:, nv:] = -3 * np.eye(2)
emat_n = emat_i.copy()
emat_n[:nv, :nv] += np.diag(.1 * np.sin(np.arange(nv)))
transitions = []
for _ in range(3):
    coefficients = np.zeros_like(emat_i)
    coefficients[:nv, nv:] = (rng.normal(size=(nv, 2)) +
                              1j * rng.normal(size=(nv, 2))) / np.sqrt(nv)
    transitions.append(coefficients)

basis_i = build_fock_basis(edrixs.FockBasisSpec.from_args(nv, ne, 2, 2))
basis_n = build_fock_basis(edrixs.FockBasisSpec.from_args(nv, ne + 1, 2, 1))
# Exact Slater-determinant ground state, requiring no assembled many-body H.
energies, orbitals = np.linalg.eigh(one_body)
initial_energy = float(energies[:ne].sum() - 6)
state = PETSc.Vec().createSeq(len(basis_i), comm=PETSc.COMM_SELF)
state_array = state.getArray()
for start in range(0, len(basis_i), 512):
    end = min(start + 512, len(basis_i))
    decoded = [basis_i.decode(index) for index in range(start, end)]
    occupied = [[bit for bit in range(nv) if decoded_state & (1 << bit)]
                for decoded_state in decoded]
    state_array[start:end] = np.linalg.det(orbitals[np.asarray(occupied), :ne])
state.normalize()

process = psutil.Process()
matrices = []
counts = Counter()
construction_seconds = [0.0]
peak_live = [0]
peak_matrix_bytes = [0.0]
events = []
initial_residual = []


def construct(builder, role):
    before = process.memory_info().rss
    start = time.perf_counter()
    matrix = builder()
    construction_seconds[0] += time.perf_counter() - start
    counts[role] += 1
    matrices.append((role, matrix))
    live = [(name, mat) for name, mat in matrices if mat.handle]
    entry_bytes = np.dtype(PETSc.ScalarType).itemsize + np.dtype(PETSc.IntType).itemsize
    matrix_bytes = sum(
        mat.getInfo()['nz_allocated'] * entry_bytes +
        (mat.getSize()[0] + 1) * np.dtype(PETSc.IntType).itemsize
        for _, mat in live
    )
    peak_live[0] = max(peak_live[0], len(live))
    peak_matrix_bytes[0] = max(peak_matrix_bytes[0], matrix_bytes)
    events.append(dict(role=role, live=len(live), rss_before=before,
                       rss_after=process.memory_info().rss,
                       estimated_live_csr_bytes=matrix_bytes))
    if role == 'hi' and not initial_residual:
        residual = state.duplicate()
        matrix.mult(state, residual)
        residual.axpy(-initial_energy, state)
        initial_residual.append(float(residual.norm()))
        residual.destroy()
    return matrix


def source(emat, left, right, role, nnz):
    builder = edrixs.build_op(
        emat, None, left, right, backend='petsc',
        backend_kws={'comm': PETSc.COMM_SELF,
                     'nnz_guess_per_row': nnz if args.preallocation == 'tailored' else None,
                     'assembly_chunk_cols': args.chunk_cols,
                     'assembly_method': args.assembly_method},
    )
    result = partial(construct, builder, role)
    result.backend = 'petsc'
    return result


hi = source(emat_i, basis_i, basis_i, 'hi', ne * (nv - ne) + 1)
hn = source(emat_n, basis_n, basis_n, 'hn', (ne + 1) * (nv - ne - 1) + 1)
trans = [source(coefficients, basis_n, basis_i, 't' + str(index), ne + 1)
         for index, coefficients in enumerate(transitions)]

options = {'matrix_lifetime': args.policy, 'nkryl': 24,
           'ksp_type': 'shifted_pminres', 'linsys_tol': 1e-10,
           'linsys_maxiter': 500}
channels = [('linear', .3, 'left', 0), ('linear', .3, 'right', 0)]
gc.collect()
baseline_rss = process.memory_info().rss
phase_peak = [baseline_rss]
stop = threading.Event()


def sample():
    while not stop.wait(.005):
        phase_peak[0] = max(phase_peak[0], process.memory_info().rss)


thread = threading.Thread(target=sample, daemon=True)
thread.start()
start = time.perf_counter()
spectrum = edrixs.rixs(
    [initial_energy], [state], hi, hn, trans, [2.75, 3.25], np.linspace(-.1, 2, 64),
    gamma_c=.3, gamma_f=.05, thin=.7, thout=.9, phi=.2,
    pol_type=channels, backend_kws=options,
)
elapsed = time.perf_counter() - start
phase_peak[0] = max(phase_peak[0], process.memory_info().rss)
stop.set()
thread.join()
maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != 'darwin':
    maxrss *= 1024
result = dict(
    policy=args.policy, assembly_method=args.assembly_method,
    preallocation=args.preallocation,
    assembly_chunk_cols=args.chunk_cols, pid=os.getpid(), nval=nv, nelectrons=ne,
    initial_dimension=len(basis_i), intermediate_dimension=len(basis_n),
    mpi_ranks=PETSc.COMM_WORLD.getSize(), petsc_version=PETSc.Sys.getVersion(),
    baseline_rss_mib=baseline_rss / 2**20,
    rixs_peak_rss_mib=phase_peak[0] / 2**20,
    process_peak_rss_mib=maxrss / 2**20,
    rixs_peak_above_baseline_mib=(phase_peak[0] - baseline_rss) / 2**20,
    elapsed_seconds=elapsed, construction_seconds=construction_seconds[0],
    constructions=dict(counts), peak_live_matrices=peak_live[0],
    peak_estimated_csr_storage_mib=peak_matrix_bytes[0] / 2**20,
    all_matrices_destroyed=all(not matrix.handle for _, matrix in matrices),
    initial_eigenvector_residual=initial_residual,
    spectrum_min=float(spectrum.min()), spectrum_max=float(spectrum.max()),
    options=options, events=events,
)
state.destroy()
np.save(args.out + '.npy', spectrum)
Path(args.out + '.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'events'}), flush=True)
