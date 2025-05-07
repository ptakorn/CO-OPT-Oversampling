# Cluster Oversampling with Optimization (CO-OPT) Algorithm

## 📌 Overview
This repository contains the Python implementation of the **Cluster Oversampling with Optimization (CO-OPT)** algorithm. CO-OPT extends the original COG method by allowing users to optimize based on any custom objective function (e.g., G-mean, F1-score, HPN) and use any scikit-learn compatible classifier. It is designed to adapt oversampling decisions dynamically for improved performance on imbalanced datasets.

The implementation supports flexible experimentation and is aligned with the methodology proposed in:
Prexawanprasut, T., & Banditwattanawong, T. (2024). *Improving Minority Class Recall through a Novel Cluster-Based Oversampling Technique*. Informatics, 11(2), 35. https://doi.org/10.3390/informatics11020035

## ⚙️ Files
- `coopt.py`: Full implementation of the CO-OPT algorithm
- `example_usage.py`: Example script showing how to run CO-OPT
- `requirements.txt`: Required Python libraries
- `LICENSE`: MIT License for use and distribution

## 🚀 Installation
```bash
git clone https://github.com/YourUsername/CO-OPT-Oversampling.git
cd CO-OPT-Oversampling
pip install -r requirements.txt

