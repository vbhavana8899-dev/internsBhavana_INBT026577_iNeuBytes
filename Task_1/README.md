# Task 1: AlexNet-style CNN Baseline vs Customized CNN (EuroSAT)

Image classification on EuroSAT RGB (27,000 satellite images, 10 land-use classes, 64x64).

## Setup
pip install -r requirements.txt

## Run
python task1_cnn.py

The script downloads the dataset, creates a frozen stratified 70/15/15 split (seed 42), trains the baseline,
runs five single-technique experiments (augmentation, batch norm, dropout, L2, cosine LR) under the same
split, seed and 20-epoch budget, builds the final CNN from only the helpful techniques, and writes everything to results/.

## Outputs (results/)
- report_draft.md: all tables and metrics for the report
- cm_baseline.png, cm_final.png: confusion matrices
- curves_baseline_vs_final.png: training curves
- *.json: raw results per run, split_seed42.npz: frozen split

## Notes
- A technique is kept only if it improves validation accuracy by >= 0.3 pp; the test set is never used for selection.
- Each run is cached; delete its json in results/ to re-run it.
