# CPU row versus COO assembly

Measured on the `COO` branch based on `37f2521`, using PETSc 3.25.5,
complex double precision, one MPI rank, and one fresh process per combination.
All runs were sequential with BLAS/OpenMP threads limited to one.
Both paths used the same entry kernels, coefficient arrays, 1024-column chunks,
exact initial state, two incident energies, two grouped outgoing channels,
24 Lanczos steps, and shifted PMINRES (`linsys_tol=1e-10`).

The row path used tailored nonzero preallocation hints. The COO path used
two entry-generation passes, exact-sized local buffers, and no pattern caching.
Times exclude JIT warmup and setup; peak RSS includes the whole process and native allocations.

| Initial / intermediate dimension | Lifetime | Row RIXS (s) | COO RIXS (s) | Row build (s) | COO build (s) | Row peak RSS (MiB) | COO peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| 48,620 / 87,516 | retain | 7.32 | 5.12 | 6.01 | 3.84 | 496 | 936 |
| 48,620 / 87,516 | recompute | 20.33 | 13.67 | 19.07 | 12.38 | 473 | 889 |
| 184,756 / 335,920 | retain | 35.70 | 24.88 | 29.58 | 18.73 | 1513 | 2387 |
| 184,756 / 335,920 | recompute | 102.73 | 68.15 | 96.51 | 61.76 | 884 | 1935 |

Largest pointwise relative spectral difference: 2.33e-10.
Both paths built five matrices with retain and 24 with recompute. All
factory-created matrices were destroyed. Recompute had one live operator
matrix at a time for either assembly method; COO buffers/mappings are
additional allocations and are included in process peak RSS.

These are CPU results, not GPU results or an assessment of cached-pattern
COO updates. Reproduce using the commands in [README.md](README.md).

## Automatic row preallocation

The main table deliberately uses accurate row preallocation, as in the
earlier lifetime benchmarks. The current automatic heuristic reserves
substantially more CSR capacity on this model. A separate check of the
48,620 / 87,516-state case gave:

| Lifetime | Default row RIXS (s) | COO RIXS (s) | Default row peak RSS (MiB) | COO peak RSS (MiB) | Tailored row peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| retain | 7.10 | 5.12 | 1407 | 936 | 496 |
| recompute | 22.02 | 13.67 | 1000 | 889 | 473 |

COO improved both time and peak RSS relative to automatic row
preallocation in this smaller case. Accurate row preallocation remained
the lowest-memory choice. The default heuristic was not changed by this experiment.
