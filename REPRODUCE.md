# REPRODUCE

Command-by-command reproduction of every number in `docs/raport.tex`.

---

## 0. Environment

```bash
python3.12 -m venv .venv && .venv/bin/python -m pip install --upgrade pip
```

```bash
.venv/bin/python -m pip install -r requirements.lock && .venv/bin/python -m pip install -e . --no-deps
```

```bash
.venv/bin/python -m pytest -q -m "not slow"
```

---

## 1. Datasets

Three generator seeds: **11, 22, 33**. Cell: `hyperplanes(n_features=20, n_hyperplanes=3, dim_hyperplanes=5)`, PCA to 5 components, split 4200/600/1200.

### 1a. Production dataset (seed 11) — already in `data/`

Only if `data/hyperplanes_*_seed11.npz` is missing:

```bash
.venv/bin/python scripts/run_freeze_hyperplanes.py
```

| flag | effect |
|---|---|
| `--scan-only` | run the 5-seed gate scan, write nothing to `data/` |
| `--skip-scan` | freeze the chosen seed without re-running the scan |

Writes `data/hyperplanes_..._seed11.npz` + `.manifest.json`. All three hashes are re-verified from the manifest after freezing.

### 1b. Seeds 22 and 33

```bash
.venv/bin/python scripts/run_main_series.py --prepare-datasets
```

Generates into `data/a7_generator_seeds/`, asserts each hash against `GENERATED_HASH_PREFIXES` in `src/qsocket/contract.py`, then exits.

**Run this before any parallel or cluster job.** Two concurrent jobs would otherwise write the same `.npz` at the same time. Afterwards start every job with `--no-generate`.


---

## 2. Main series

One command runs all four stages in order; each stage can stop the next.

```bash
.venv/bin/python scripts/run_main_series.py \
    --dataset-seeds 11 --out-dir outputs/main/ds11 --no-generate --workers 10
```

```bash
.venv/bin/python scripts/run_main_series.py \
    --dataset-seeds 22 --out-dir outputs/main/ds22 --no-generate --workers 10
```

```bash
.venv/bin/python scripts/run_main_series.py \
    --dataset-seeds 33 --out-dir outputs/main/ds33 --no-generate --workers 10
```

**One `--out-dir` per dataset seed.** Resume reads the CSV in that directory; two jobs sharing one directory interleave writes into one CSV.

### Stages

| # | stage | what it does | stops the run on |
|---|---|---|---|
| 0 | validate | dataset hashes, θ pairing A↔B, column schema | any usage error, before training |
| 1 | lr | `select_lr` per (dataset × dilution × ansatz) over arms A and B, seeds 1–3, contract grid; arm E on the grid + one point | — |
| 2 | gates | G1 and G2 per generator seed at the selected lr | gate failure |
| 3 | main | the grid; `test` and `val` as separate **rows** | — |
| 4 | summary | per-arm means and σ, the estimands, θ diagnostics, budget hits | — |

Stop early with `--stop-after lr` or `--stop-after gates`.

### Grid

| axis | values |
|---|---|
| dilution | `linear` `h2` `h4` `h42` (6 / 15 / 29 / 295 head params) |
| ansatz | `L1` `L2` (arm F uses `product`) |
| training seed | 1–10 |
| lr selection seeds | 1–3 |
| R | 2 |
| arms | A, F trained · B, E, D_matched, D_best frozen (feature cache) |

Arms E, F, D_matched, D_best carry **no ansatz dimension**; D_best carries no dilution dimension. That is why rows are missing by construction.

### Outputs per `--out-dir`

| file | content |
|---|---|
| `a7_results.csv` | one row per (arm × cell × seed × split) — **the raw data of the study** |
| `a7_lr_table.csv` | full lr × arm table, not just the winner |
| `a7_lr_selection.json` | selected lr per cell + where the argmax sits |
| `a7_gates.json` | G1/G2 records |
| `a7_summary.json` | descriptive statistics |
| `predictions/` | per-test-row correctness — **required for McNemar, cannot be recomputed later** |
| `weights/` | trained θ and head state dict, < 2 kB per run |

### Smoke test first

```bash
.venv/bin/python scripts/run_main_series.py --dry-run --no-generate --out-dir outputs/dry_run
```

2 seeds × 1 dilution × 2 ansatze × every arm, 1 lr seed. 

---

## 3. Combine the three datasets

`run_a8_analysis.py` pools the generator seed as a fixed effect and needs one CSV. Nothing in the repo writes it — concatenate by hand:

```bash
{ head -1 outputs/main/ds11/a7_results.csv; \
  for d in ds11 ds22 ds33; do tail -n +2 outputs/main/$d/a7_results.csv; done; } \
  > outputs/stats/main/a7_results_combined.csv
```

Expected: 620 + 660 + 660 = **1940** rows.

---

## 4. Analysis — the report's numbers

```bash
.venv/bin/python scripts/run_a8_analysis.py \
    --results outputs/stats/main/a7_results_combined.csv \
    --predictions-dir outputs/main/ds11/predictions \
                      outputs/main/ds22/predictions \
                      outputs/main/ds33/predictions \
    --out-dir outputs/stats/main
```

**Pass the predictions directory of every run.** A combined CSV with one predictions directory computes McNemar on a third of the data.

| flag | effect |
|---|---|
| `--skip-slow-verification` | skip the G1 and ceiling recomputation; they then report `NOT VERIFIED`, never `passing` |
| `--no-figures` | tables only |
| `--probe` | analyse a correlator-probe directory instead |


### What it computes

- Δ_AB (the confirmatory question), Δ_AE with its mandatory decomposition, Δ_AF, Δ_BD_matched, Δ_BD_best
- replication across the three generator seeds side by side, then pooled with the generator seed as a **fixed** effect (30 blocked paired differences)
- exact sign test and exact Wilcoxon, plus t for comparison; MixedLM as a check; TOST for Δ_AE only
- the three uncertainty accounts side by side: CI over the paired differences, binomial SE on 1200 test rows, McNemar on discordant pairs

### Outputs

| path | content |
|---|---|
| `tables/estimands.csv` | the Δ values and their intervals |
| `tables/verdicts.csv` | verdict per cell against the pre-declared table |
| `tables/replication.csv` | per-dataset side by side vs pooled |
| `tables/diagnostics.csv` | σ, MDE, seeds-needed, θ diagnostics |
| `tables/raport_verification.csv` | recomputes values quoted in report and fails on mismatch |
| `tables/arms.csv` | per-arm absolute accuracies |
| `tables/00_provenance.csv` | git commit and env hash **of the rows**, not of the current tree |
| `a8_summary.json` | everything above, machine-readable |
| `figures/*.pdf` | fig1a, fig1b, fig2, fig3, fig4, fig5 |

An incomplete grid is refused as a result — tables are stamped `PROVISIONAL — NOT A RESULT`.

---

## 5. Supplementary tables

```bash
.venv/bin/python scripts/run_supplementary.py \
    --main outputs/stats/main/a7_results_combined.csv \
    --probe outputs/h3_probe/a7_results.csv \
    --lr-table outputs/main/ds11/a7_lr_table.csv \
    --lr-table outputs/main/ds22/lr_table.csv \
    --lr-table outputs/main/ds33/a7_lr_table.csv \
    --weights-dir outputs/h3_probe/weights \
    --out-dir outputs/supplementary
```

**`--probe`, `--lr-table` and `--weights-dir` are `action="append"`.

`--skip-slow` drops the Fourier spectrum and the head-init Monte Carlo.

Writes to `outputs/supplementary/tables/`: `degeneracy.csv`, `diagnostics.csv`, `ridge_contrast.csv`, `probe_estimands.csv`, `head_init.csv`, `fourier_support.csv`, `data_rank.csv`, `shared_lr_bias.csv`, `displacement.csv`.

---

## 6. The σ / power figure

```bash
.venv/bin/python scripts/plot_sigma_power.py --tables outputs/stats/main/tables
```

Reads `estimands.csv` and `diagnostics.csv`, writes `outputs/stats/main/figures/fig_sigma_power.pdf`. Draws what the pipeline decided; it does not recompute the estimands.

---

## 7. Probes

### 7a. `h3` head, off the dilution axis

```bash
.venv/bin/python scripts/run_main_series.py \
    --dilutions h3 --out-dir outputs/h3_probe --no-generate --workers 10
```

### 7b. Correlator readout (order = 2, 15 observables)

```bash
.venv/bin/python scripts/probe_readout_order.py --dry-run
```

```bash
.venv/bin/python scripts/probe_readout_order.py \
    --dataset-seeds 11 --out-dir outputs/probe_corr/ds11 --workers 10
```

Repeat for 22 and 33. `--ansatz` is default L1 only, to change that `--ansatz L1 --ansatz L2`.

Analyse:

```bash
.venv/bin/python scripts/run_a8_analysis.py --probe \
    --results outputs/probe_corr/ds11/probe_results.csv \
    --out-dir outputs/probe_corr/analysis_ds11
```

### 7c. Is the lr grid clipped too low?

```bash
caffeinate -dimsu .venv/bin/python scripts/probe_d21_lr_edge.py \
    --phase1-dir outputs/main --out-dir outputs/lr_edge --workers 10
```

Verdict: WIDEN if arm A's argmax sits at the probe point in ≥ half the cells **and** the median gain exceeds 0.020. Otherwise LEAVE. Rule fixed before the measurement.

### 7d. Gradient SNR at initialisation

```bash
.venv/bin/python scripts/probe_gradient_snr.py
```

`SNR_i = |mean_b(g_ib)| / std_b(g_ib)` over 8 batches, at arm A's initialisation. No optimiser step is taken. Writes `outputs/p1_gradient_snr/`.

Superseded by the main series. Writes `outputs/a4_pilot/`.

### 7e. Expressibility / entanglement of the ansatze

```bash
.venv/bin/python scripts/run_expressibility.py
```

---

## 8. Hardware (IQM Spark / ODRA 5) - Not tested yet

```bash
.venv/bin/python scripts/run_on_hardware.py --url <IQM_URL> --shots 4096 \
    --out outputs/hardware/rows.csv --note "session label"
```

`calibration_set_id` is mandatory: `qsocket.results.append_result_row` rejects a hardware row without it and writes nothing, because σ_hw cannot then be decomposed.

---


`env_hash` differs between macOS and Linux (the CPU torch wheel carries `+cpu`). That is recorded, not repaired; the claim that must hold is agreement of `<Z_i>` at 1e-10 across the three implementations.
