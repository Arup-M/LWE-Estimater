# LWE Primal Attack Estimator with BKZ Profile Visualization

This repository provides a Python implementation for estimating the cost of the **primal lattice attack on LWE** (Learning With Errors) instances.  
It is adapted from the [CRYSTALS security-estimates](https://github.com/pq-crystals/security-estimates) codebase, with added **interactive visualization of BKZ basis profiles** to better understand how BKZ reduction affects Gram–Schmidt lengths.

---

## 📌 Features
- Implements standard lattice attack cost models:
  - Root-Hermite factor `δ_b` for BKZ
  - Classical SVP cost estimates
- Simulates **BKZ-reduced basis shapes** (`construct_BKZ_shape`)
- Computes **primal attack cost** on LWE instances
- Performs **parameter search/optimization** over:
  - Number of samples `m`
  - Blocksize `b`
- Provides **live plots** of BKZ profiles as the optimizer finds better attacks

---

## ⚙️ Installation
The code requires **Python 3.8+** with `numpy` and `matplotlib`:

```bash
pip install numpy matplotlib
