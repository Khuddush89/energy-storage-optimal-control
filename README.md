# Energy Storage Optimal Control 🔋⚡

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![SciPy](https://img.shields.io/badge/SciPy-Optimization-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow)

A Python framework for solving optimal control problems in multi-timescale energy storage systems using direct optimization methods. This repository implements model predictive control strategies to optimize battery storage operations under various grid scenarios.

## 📖 Overview

Modern energy grids require sophisticated control strategies to manage battery storage systems effectively. This project provides a computational framework for determining optimal charging and discharging schedules that minimize operational costs while satisfying physical constraints and grid requirements.

The implementation uses direct optimization techniques to solve the optimal control problem, providing actionable insights for energy storage operation across multiple scenarios including peak shaving, grid support, and base load operations.

## 🚀 Key Features

- **🎯 Direct Optimization**: Implementation using SciPy's SLSQP algorithm with constraint handling
- **🔋 Multiple Operational Scenarios**:
  - **Base Case**: Standard grid operation with constant demand
  - **Peak Shaving**: Reduce demand during morning and evening peaks
  - **Grid Support**: Provide ancillary services with stochastic demand
- **⚡ Realistic System Dynamics**: Battery degradation, efficiency losses, and power limits
- **📊 Comprehensive Visualization**: State trajectories, control strategies, and performance metrics
- **💾 Data Export**: CSV outputs for further analysis in external tools
- **📈 Performance Analysis**: Cost comparisons and optimization effectiveness metrics

## 🏗️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/yourusername/energy-storage-optimal-control.git
cd energy-storage-optimal-control

# Install dependencies
pip install -r requirements.txt
