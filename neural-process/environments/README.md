# Environment Pins

Two captured snapshots of the software environments this repository runs in.
Each is a point-in-time pin, not a `requirements.txt` you should install
into a fresh venv — the versions come from whatever was installed on the
respective machines when the runs were captured.

| Directory | Purpose | What it pinned |
|---|---|---|
| `hprc-grace/` | **Production** — Texas A&M HPRC Grace cluster | Python + R versions used for LOO validation runs and the final multi-modal production sweep. This is the environment that produced every reported metric (LOO CV and the CV0/CV00/CV1/CV2 sweeps). |
| `local-macbook/` | **Development / debugging** — developer workstation | Python + R versions used for writing code, local CPU debugging, and smoke runs. *Not* used for any published metric or benchmark. |

Both pins are for the same repository, same commit; the differences are
natural drift between a developer laptop (Python 3.13 + PyTorch 2.6 +
macOS / MPS) and the managed HPC stack (Python 3.11 + PyTorch 2.5 +
RHEL / CUDA). Read `hprc-grace/README.md` for the module stack that
has to be loaded before the Python venv will even launch, and
`local-macbook/README.md` for the known drifts vs. Grace.

If something works locally but fails on Grace (or vice-versa), check the
**Known differences** table in `local-macbook/README.md` first — most
symptoms trace back to `numpy` / `scipy` / `torch` minor-version drift.
