# A Two-Stage Fuzzy Forecasting and PSO-Based Dispatch Framework for Grid-Connected Microgrid with BESS and V2G Electric Vehicle

Code and data accompanying the manuscript (Md. Alamgir Hossain, Md. Shahjahan, Khulna University of Engineering & Technology). This repository reproduces every table, figure, and statistical test reported in the paper from the released data and fixed random seeds.


[![DOI](https://zenodo.org/badge/1304791112.svg)](https://doi.org/10.5281/zenodo.21427697)

## Repository contents

| File | Description |
|---|---|
| `fcm_tsk_pso_de_EV_V2G_multiday_FFNN_LSTM_V1.m` | Main MATLAB pipeline: FCM-TSK / FFNN / LSTM forecasting (Stage A), PSO/DE dispatch optimisation (Stage B), single-day and 20-day multi-day evaluation, all statistical tests (Diebold–Mariano, Wilcoxon, one-sample *t*-test), and figure generation. |
| `Microgrid_data.csv` | Raw, unmodified PV/load/price data as downloaded from the [FLYao District Microgrid Dataset](https://github.com/FLYao123/District-microgrid-dataset) (Yao et al., 2025), included for independent provenance verification. Not read directly by the pipeline. |
| `Microgrid_with_EV.csv` | The processed input file actually consumed by the pipeline: PV, load, and price (converted W→kW), the two auxiliary channels released with the source dataset (`Unmeet_kWh_`, `CI_gco2_kWh_`), and four auxiliary EV-related feature columns (see **Data notes** below). |
| `results.log` | Full console output (MATLAB `diary`) from the run that produced every number reported in the paper. |
| `multiday_results_20days_10runs.csv` | Per-day forecast MAE and dispatch cost/savings table for all 20 evaluated test days (10 matched-seed runs per day), auto-saved by the pipeline. |

## Requirements

- MATLAB R2021a or later
- **Deep Learning Toolbox** (required for the LSTM benchmark, `lstmLayer`)
- Statistics and Machine Learning Toolbox (`ranksum`, `signrank`, `ttest`, `fitcsvm`-family utilities if used)
- No GPU required; the LSTM benchmark will use a GPU automatically if MATLAB detects one (`'ExecutionEnvironment','auto'`). See **Reproducibility note** below.

## How to run

1. Place `Microgrid_with_EV.csv` in the same folder as the `.m` script (or update `CSV_FILE` at the top of the script).
2. Open MATLAB in that folder and run:
   ```matlab
   fcm_tsk_pso_de_EV_V2G_multiday_FFNN_LSTM_V1
   ```
3. The script runs, in order:
   - Data loading, feature construction, and the FCM-TSK / FFNN / LSTM forecaster training (Stage A);
   - A single representative-day forecast and dispatch run, reproducing Tables 2–4 and Figures 1–9;
   - The 20-day multi-day evaluation (`MULTIDAY_N_DAYS = 20`, `MULTIDAY_N_RUNS = 10`), reproducing Tables 5–8 and Figures 10–11.
4. To capture the full console log as in `results_log.txt`, wrap the run with:
   ```matlab
   diary results_log_rerun.txt
   fcm_tsk_pso_de_EV_V2G_multiday_FFNN_LSTM_V1
   diary off
   ```

Typical runtime: a few minutes for the single-day section; the 20-day campaign (200 PSO runs + 200 DE runs, 10 matched seeds × 20 days) takes substantially longer — expect on the order of tens of minutes to a few hours depending on hardware.

## What reproduces what

| Paper item | Source |
|---|---|
| Table 2, Table 3, Figures 1–3 | Single-day section (first ~half of script output) |
| Table 4, Figures 4–9 | Single-day PSO/DE dispatch section |
| Tables 5–8, Figures 10–11 | Multi-day evaluation section (`MULTI-DAY EVALUATION` block in the log) |
| Abstract / Conclusion headline numbers | Multi-day evaluation summary block at the end of the log |

## Data notes

The source dataset (Yao et al., 2025, derived from Chong Aih's measured U.S. microgrid record) contains **no EV telemetry**. `Microgrid_with_EV.csv` therefore includes four auxiliary columns — `EV_Charging_kW`,
`EV_Discharging_kW`, `EV_SoC`, `Vehicle_Count` — that are used **exclusively as lagged input features** for the Stage-A forecasters and play **no role** in defining the Stage-B dispatch scenario. They are clipped to the
single-vehicle charger rating before use and are never used as a forecast target.

The actual single-EV dispatch scenario optimised in Stage B (40 kWh battery, 35% arrival SoC, 70% required departure SoC, 3.6 kW practical AC charging/discharging limit, contracted 2.0 kWh V2G service window) is
specified **independently** via fixed parameters inside the MATLAB script (`EV_params`, printed at the top of the log and listed in the paper's System Parameters table) and is unaffected by the values of the four
auxiliary columns above. See Section 3 of the manuscript for the full discussion.

## Reproducibility note

All FCM-TSK, FFNN, and dispatch (PSO/DE) results are **exactly reproducible** from the released code, data, and fixed random seeds. This was independently verified by running the full pipeline twice and diffing
the console output. The single exception is the **LSTM benchmark**, which by default trains on GPU hardware if available; GPU floating-point scheduling is not bit-reproducible across runs, so LSTM-specific MAE values
and the corresponding Diebold–Mariano statistics may vary marginally between executions. To obtain fully deterministic results including the LSTM, set `'ExecutionEnvironment','cpu'` in the LSTM training options near the top of the script.

## Citation

If you use this code or data, please cite:

```
[Full citation to be added upon publication]
```

and the underlying dataset:

```
F. Yao, W. Zhao, M. Forshaw, Y. Song, "A holistic power optimization approach for microgrid control based on deep reinforcement learning," Neurocomputing, 2025, 131375. Dataset: https://github.com/FLYao123/District-microgrid-dataset Chong Aih, "Microgrid reinforcement learning dataset (US microgrid hourly load, market price, and PV)," GitHub repository, 2021.
```

## License

Code in this repository is released under CC BY 4.0 (see `LICENSE`). The redistributed raw data file (`Microgrid_data.csv`) originates from the FLYao District Microgrid Dataset repository; please consult that repository's own license terms before further redistribution.


## Contact

Md. Alamgir Hossain — mah@eee.kuet.ac.bd
Dept. of EEE, Khulna University of Engineering & Technology, Bangladesh
