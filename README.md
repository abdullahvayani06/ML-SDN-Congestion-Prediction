# ML-Based Congestion Prediction for SDN

Proactive multi-path routing and aggregate-load-aware recovery using Random Forest and Gradient Boosting, Ryu, OpenFlow 1.3 and Mininet.

## Structure

- `src/controllers/`: baseline and proactive Ryu controllers.
- `src/network/`: diamond topology and traffic workload.
- `src/ml/`: synthetic episode generation and training.
- `src/evaluation/`: evaluation of telemetry, iPerf and NIC counters.
- `models/`: original trained model bundle.
- `results/recorded/`: original CSV, JSON and log files, with filenames preserved for evaluation.
- `results/figures/`: original results image.
- `docs/`: supplied research paper.
- `scripts/`: new launcher; `legacy/` contains original shell scripts for reference.

## Environment

The paper reports Ubuntu 22.04, Python 3.10.12, Mininet 2.3.0, OVS 2.17 and Ryu 4.34. Use a dedicated Ubuntu VM with Mininet, Open vSwitch, iPerf **v2**, ping and sudo installed. `requirements.txt` lists Python dependencies; it is not a fully pinned environment lock. Ryu may require compatible versions of its dependencies. Match the original scikit-learn version when loading the supplied pickle, or retrain it. Only load trusted pickle files.

## Run from the project root

```bash
# Analyze recorded data without running Mininet
bash scripts/run.sh evaluate
# Train in a fresh run directory
bash scripts/run.sh train
# Run individual configurations (Ubuntu VM)
bash scripts/run.sh baseline
bash scripts/run.sh ml
# Run all ten configurations, with three trials for the five repeated modes
bash scripts/run.sh all
```

The launcher stages unmodified source files and the model in a fresh directory under `results/runs/` because the original code relies on relative filenames. Training outputs stay in that run directory; the `all` action trains before running. Recorded results and source files are not overwritten. Individual runs use the supplied model. The full suite checks both iPerf and NIC sent-packet totals against 44,800 before accepting a trial. Run live experiments only in a dedicated VM: Mininet cleanup affects existing Mininet state. The original evaluator prints tables; the supplied figure is archived separately.

## Legacy scripts

The original scripts are retained unchanged for provenance, not as recommended entrypoints. Several rewrite Python files, use hard-coded home-directory paths or older acceptance criteria. Use the new launcher instead. Original Python code, model and recorded data are unchanged; only cache files and a temporary editor lock were removed. Live network execution requires the original networking environment and has not been validated in this packaging environment.
