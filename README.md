# ML Blueprint

Reusable ML toolbox for:
- Tabular ML (scikit-learn)
- Tabular DL (PyTorch MLP)
- EDA reports
- Statistical methods (A/B, CI, permutation/bootstrap)

## Quickstart

### 1) Create env & install (editable)
```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"