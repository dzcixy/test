# HyperWafer SpGEMM Communication Simulation

> End-to-end pipeline for **hypergraph-based SpGEMM communication analysis** on wafer-scale style networks, with integration to **Mt-KaHyPar** and **ASTRA-sim**.

This repository contains a reproducible pipeline to evaluate the communication
behavior of sparse matrix–matrix multiplication (SpGEMM) under different
task-to-tile mappings on wafer-style 2D meshes.

The code is designed to compare:

- **WAFERSPMM** – a row-block baseline mapping.
- **HyperWafer** – a hypergraph-based mapping that co-locates tasks sharing
  input rows of \(B\), aiming to reduce communication volume and link hotspots.

The pipeline integrates:

- SuiteSparse Matrix Collection (via `ssgetpy`)  
- Hypergraph partitioning (via [Mt-KaHyPar](https://github.com/kahypar/mt-kahypar))  
- A 2D mesh oracle for exact per-link communication simulation  
- MICRO-level workload generation for [Chakra](https://github.com/astra-sim/chakra)  
- Analytical network simulation via [ASTRA-sim](https://github.com/astra-sim/astra-sim)

---

## 1. Overview

### 1.1 High-level flow

The end-to-end pipeline is:

1. **Matrix intake**
   - Download a sparse matrix \(A\) from the SuiteSparse Matrix Collection using `ssgetpy`.
   - Construct \(B\). By default we use \(B = A^T\), but you can plug in your own \(B\).

2. **Hypergraph abstraction of Gustavson SpGEMM**
   - We target Gustavson-style SpGEMM \(C = A 	imes B\) with row-wise expansion.
   - Tasks: each row \(A_{i,:}\) is a Gustavson row task.
   - Hypergraph:
     - Vertices: rows of \(A\) (SpGEMM row tasks).
     - Hyperedges: columns of \(A\); each hyperedge connects all rows that reuse
       the same row \(B_{k,:}\).
     - Hyperedge weights: nnz of \(B_{k,:}\) (indicating communication cost).

3. **Partitioning & mapping onto a 2D mesh**
   - Baseline (**WAFERSPMM**):
     - Row-block mapping: partition rows of \(A\) into equal-sized contiguous blocks,
       assign each block to a tile.
   - Hypergraph mapping (**HyperWafer**):
     - Use Mt-KaHyPar to partition the hypergraph into `num_parts` blocks.
     - Target graph is a 2D mesh (e.g., 8×8 for 64 tiles), encoded in METIS format.
   - For each mapping, we also compute:
     - For each row \(B_{k,:}\), the **owner tile** (the tile with the majority of
       tasks consuming that row).

4. **2D mesh point-to-point oracle**
   - We model the wafer network as a 2D mesh with dimension-ordered routing
     (Manhattan paths).
   - For each row \(B_{k,:}\):
     - Determine the set of tiles that need this row (based on nonzeros in column \(k\) of \(A\)).
     - Use the owner tile as the source.
     - For each destination tile, route a unicast along the 2D mesh and accumulate:
       - Per-link traffic (bytes)
       - Total bytes
       - GB-hop (= bytes × hops)
   - This oracle is **fully point-to-point**, not a collective abstraction.

5. **MICRO workload generation for AstraSim**
   - From the oracle, we compute the bytes contributed by each \(B\)-row.
   - We then batch rows into **MICRO phases**, each represented as an
     equivalent `ALLGATHER` with `comm_size = phase_bytes`:
     - Mode `single`: one big ALLGATHER per mapping.
     - Mode `auto_rows`: choose rows-per-layer so WAFERSPMM has ≤ `micro_max_layers`
       phases; HyperWafer uses the same rows-per-layer and thus typically has far
       fewer phases (reflecting reduced cross-tile reuse).
     - Mode `rows_manual`: use a fixed `rows_per_layer` (e.g., 512) for both mappings.
   - We emit Chakra-compatible `MICRO` text workloads.

6. **Chakra & ASTRA-sim integration**
   - Use `chakra_converter` to convert each MICRO text file into ET workloads.
   - Run `AstraSim_Analytical_Congestion_Aware` on both:
     - WAFERSPMM workload
     - HyperWafer workload
   - Parse per-system communication times and report:
     - The maximum communication time over all systems (NPUs) for each mapping.

7. **Final comparison**
   - For both WAFERSPMM and HyperWafer, we report:
     - Oracle total bytes
     - Oracle GB-hop
     - Oracle peak-link load (bytes on the hottest link)
     - MICRO total bytes
     - MICRO number of layers
     - AstraSim communication time
   - And derived ratios:
     - Communication volume reduction
     - GB-hop reduction
     - Peak-link reduction
     - Communication time speedup (AstraSim)

---

## 2. Repository layout

A typical layout of this repository is:

```text
.
├── src/
│   └── spgemm_pipeline.py     # main Python CLI
├── external/
│   └── astra-sim/             # fork of ASTRA-sim as a git submodule
├── examples/
│   └── run_dimacs10_delaunay_n19.sh
├── scripts/
│   └── build_astrasim_analytical.sh
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

> Note: the actual structure may evolve over time. See this README and the example
> scripts for the most up-to-date usage.

---

## 3. Requirements

### 3.1 Python

- Python 3.9+
- `numpy`
- `scipy`
- `ssgetpy`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 3.2 External tools (not bundled as Python packages)

You need the following external tools installed and built:

1. **Mt-KaHyPar**

   - GitHub: https://github.com/kahypar/mt-kahypar  
   - Build the `MtKaHyPar` binary and point `--mtk-bin` to it.

2. **Chakra converter (`chakra_converter`)**

   - Part of the Chakra / ASTRA-sim ecosystem.  
   - Make sure the `chakra_converter` CLI is in your `PATH`, or pass its path via `--chakra-bin`.

3. **ASTRA-sim (analytical backend)**

   - This repository can include a **fork of ASTRA-sim** as a git submodule under:
     `external/astra-sim/`.

   - We keep the original MIT license and add minimal modifications needed for our experiments.

   - You can fetch the submodule after cloning:

     ```bash
     git submodule update --init --recursive
     ```

   - Then build the analytical backend (see below).

---

## 4. Building ASTRA-sim from the submodule

If you use the vendored ASTRA-sim submodule, you can compile the analytical backend via:

```bash
./scripts/build_astrasim_analytical.sh
```

This script will:

- Enter `external/astra-sim/`
- Invoke `./build/astra_analytical/build.sh`
- Produce `AstraSim_Analytical_Congestion_Aware` under:

```text
external/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware
```

You can override this path at runtime via the `--astrasim-bin` argument.

---

## 5. Quick start

### 5.1 Clone and fetch submodules

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
git submodule update --init --recursive
```

### 5.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5.3 Build ASTRA-sim analytical backend

```bash
./scripts/build_astrasim_analytical.sh
```

### 5.4 Run the example

Edit `examples/run_dimacs10_delaunay_n19.sh` to set the correct paths for:

- `MTK_BIN`
- `SYSTEM_CONFIG`
- `NETWORK_CONFIG`
- `REMOTE_MEM_CONFIG`

Then run:

```bash
bash examples/run_dimacs10_delaunay_n19.sh
```

You should see logs from:

- The hypergraph construction and Mt-KaHyPar partitioning
- The 2D mesh oracle (reporting total bytes, GB-hop, avg tiles per B-row, etc.)
- MICRO workload generation (WAFERSPMM vs HyperWafer)
- Chakra conversion
- AstraSim analytical backend

At the end, a summary similar to the following will be printed:

```text
========== SUMMARY (PIPELINE v8) ==========
Matrix: DIMACS10/delaunay_n19
A: shape=(524288, 524288), nnz=3145646
B: shape=(524288, 524288), nnz=3145646
num_parts: 64
micro_mode: auto_rows
effective rows_per_layer used: 512

WAFERSPMM baseline:
  Oracle total_bytes      = ...
  Oracle GB-hop           = ...
  Oracle peak-link_bytes  = ...
  MICRO total_bytes       = ...
  MICRO num_layers        = ...
  AstraSim Comm time      = ...

HyperWafer mapping:
  Oracle total_bytes      = ...
  Oracle GB-hop           = ...
  Oracle peak-link_bytes  = ...
  MICRO total_bytes       = ...
  MICRO num_layers        = ...
  AstraSim Comm time      = ...

Volume reduction (WAFERSPMM / HyperWafer):      ...
GB-hop reduction (WAFERSPMM / HyperWafer):      ...
Peak-link reduction (WAFERSPMM / HyperWafer):   ...
Comm-time speedup (AstraSim, WAFERSPMM/HyperWafer): ...
===========================================
```

---

## 6. CLI usage

The main entry point is:

```bash
python src/spgemm_pipeline.py [arguments...]
```

Key arguments:

- `--matrix-selector`  
  SuiteSparse matrix selector. Supports:
  - `1234` → exact id
  - `Group/Name` → group/name
  - `foo` → substring search on matrix name

- `--num-parts`  
  Number of tiles / NPUs / partitions (e.g., 16, 64). Also equals the number of
  nodes in the 2D mesh.

- `--mtk-bin`  
  Path to the `MtKaHyPar` executable.

- `--astrasim-bin`  
  Path to `AstraSim_Analytical_Congestion_Aware`.

- `--system-config`, `--network-config`, `--remote-mem-config`  
  System / network / remote memory configs for AstraSim.

- `--chakra-bin`  
  `chakra_converter` binary name or full path.

- `--micro-mode`  
  Controls how we batch B-rows into MICRO phases:
  - `single`: one ALLGATHER per mapping.
  - `auto_rows`: **recommended**; automatically chooses rows-per-layer so
    WAFERSPMM has at most `micro_max_layers` phases, HyperWafer uses the same
    rows-per-layer.
  - `rows_manual`: fixed `--micro-rows-per-layer` rows per phase.

- `--micro-max-layers`  
  Used in `auto_rows` mode as the target upper bound on WAFERSPMM layers.

- `--micro-rows-per-layer`  
  Used in `rows_manual` mode.

- `--workdir`  
  Directory to store intermediate outputs (SuiteSparse matrices, hypergraphs,
  partitions, MICRO workloads, ET traces, logs, etc.).

Run `python src/spgemm_pipeline.py -h` to see all available options.

---

## 7. Methodology notes

- **Oracle is fully point-to-point**  
  The 2D mesh oracle counts real point-to-point multi-cast traffic induced by
  Gustavson SpGEMM—each row \(B_{k,:}\) is routed from its owner tile to all
  tiles that need it, via Manhattan paths.

- **MICRO uses collectives as an abstraction**  
  For AstraSim, we compress the per-\(B\)-row traffic into equivalent
  ALLGATHER phases. This is an intentional modeling choice: the oracle is used
  for precise communication metrics; AstraSim is used for relative wall-clock
  trends under a standard collective-based runtime.

- **Hypergraph weights reflect communication pressure**  
  Hyperedges are weighted by the nnz of the corresponding row of \(B\), so
  Mt-KaHyPar naturally prefers to co-locate tasks sharing "expensive" rows of \(B\).

---

## 8. License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

The vendored / forked ASTRA-sim code under `external/astra-sim/` retains the
original license and copyright notices from the ASTRA-sim authors.

---

## 9. Acknowledgements

This project builds on the following open-source components:

- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
- [ssgetpy](https://github.com/bkj/ssgetpy)
- [Mt-KaHyPar](https://github.com/kahypar/mt-kahypar)
- [Chakra](https://github.com/astra-sim/chakra)
- [ASTRA-sim](https://github.com/astra-sim/astra-sim)

If you use this repository in academic work, please also consider citing the
original ASTRA-sim and Mt-KaHyPar papers.