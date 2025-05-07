# CO-OPT: Cluster Oversampling with Optimization

**CO-OPT** is an advanced oversampling algorithm designed to improve classification on imbalanced datasets by combining cluster-wise SMOTE with an objective-driven optimization loop. This method allows users to plug in any scikit-learn-compatible classifier and objective metric (e.g., G-mean, F1-score, HPN).

## 📌 Key Features
- Objective function flexibility (custom or built-in metrics)
- Patience-based early stopping
- Scikit-learn classifier support
- Works with numerical CSV datasets

## 🧩 Requirements
- Python 3.x
- `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib` (optional), `scipy` (optional)

Install using:
```bash
pip install -r requirements.txt
