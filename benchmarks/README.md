# PETSc assembly comparison

`petsc_assembly.py` runs the same single-rank many-body RIXS problem using row
insertion or bulk COO assembly, with either matrix lifetime policy. Both
construction options produce AIJ (CSR) matrices. The benchmark uses a
noninteracting model with an exact Slater-determinant initial state, two incident
energies, two outgoing channels sharing one incoming polarization, 24 Lanczos
steps, and shifted PMINRES. No operator matrices are constructed before timing.
A tiny matrix warms the Numba kernel before the calculation.

Use the repository's PETSc environment, with Numba and psutil installed. Run each
combination in its own process, sequentially to avoid CPU/memory contention:

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/petsc_assembly.py recompute --assembly-method row \
  --nval 18 --out /tmp/row-recompute
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python benchmarks/petsc_assembly.py recompute --assembly-method coo \
  --nval 18 --out /tmp/coo-recompute
```

Repeat with `retain`. `--nval 18` gives dimensions 48,620 and 87,516;
`--nval 20` gives 184,756 and 335,920. Both methods use identical coefficient
arrays and `--chunk-cols` (default 1024). Row assembly uses tailored preallocation
hints; use `--preallocation default` to measure the automatic row preallocation
heuristic instead. COO determines the pattern itself. COO includes both entry-generation
passes and preallocation in its construction time, with no pattern caching.

Each run writes a JSON measurement and an NPY spectrum. Compare spectra with
`numpy.testing.assert_allclose`. JSON includes total RIXS time, construction
time, baseline RSS, OS peak RSS (including native memory), live matrix counts,
and construction events. `peak_estimated_csr_storage_mib` estimates only the
CSR arrays; it excludes COO mappings and other PETSc internals. GPU performance
is not measured by this CPU benchmark.
