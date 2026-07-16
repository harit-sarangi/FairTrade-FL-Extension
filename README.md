# FairTrade: Achieving Pareto-Optimal Trade-offs Between Balanced Accuracy and Fairness in Federated Learning

This repository is an extended, fully debugged fork of the **FairTrade** framework published in the *Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI-24)*. 

As Federated Learning (FL) environments inherently suffer from class imbalances and statistical heterogeneity across distributed clients, optimizing for standard accuracy can yield misleading fairness evaluations. FairTrade is a multi-objective optimization (MOO) framework leveraging Multi-Objective Bayesian Optimization (MOBO) via BoTorch to dynamically negotiate Pareto-optimal trade-offs between **balanced accuracy** and **fairness**.

---

## Summary of Key Changes & Modifications

This fork introduces crucial framework fixes, extends the evaluation pipeline, and incorporates joint multi-attribute optimization support:

### 1. Framework Bug Fixes (Task 4)
* **Resolved Loop Assignment Crash (Bug A):** Fixed a fatal `ValueError: not enough values to unpack (expected 3, got 1)` in `FairTrade.py` (and `FairTrade-crypten.py`) where the training loop tried to unpack a single 2-element objective tensor into three variables.
* **Fixed Non-existent Causal Class Call (Bug B):** Patched a major `NameError` in `constraint.py` where the constructor and forward passes of `AverageTreatmentEffectLoss` attempted to invoke `super(EqualOpportunityLoss, self)` instead of targeting its actual class space.

### 2. Intersectional Audit Module (Task 2)
* Added a standalone pipeline (`evaluate_intersection.py`) to systematically audit compound, multi-dimensional bias. It computes selection rates and Statistical Parity Differences (SPD) across both isolated individual features (gender, race) and their respective intersectional subgroups (e.g., White Males, Non-White Females).

### 3. Minimax Joint Multi-Attribute Fair FL (Task 3)
* Extended the core learning pipeline to optimize jointly for **gender** and **race**. 
* Designed a **Minimax (Max-SPD)** formulation to preserve the efficiency of the 2D BoTorch hypervolume Pareto search:
  $$\min_{\theta} \quad \mathcal{F}(\theta) = \max \left( \left\vert{}\text{SPD}_{\text{gender}}(\theta)\right\vert{}, \left\vert{}\text{SPD}_{\text{race}}(\theta)\right\vert{} \right)$$
* Updated \texttt{load\_data\_utilities.py}, \texttt{constraint.py}, and \texttt{FairTrade.py} to extract both protected features, compute concurrent constraint violations, and dynamically backpropagate the worst-case demographic bottleneck.

---

## Repository File Structure

```text
FairTrade/
│
├── datasets/                 # Preprocessed and raw datasets (Adult, Bank, etc.)
│── FairTrade_pre_changes.py  # Original codebase entry point for referring code pre task changes/BO orchestration pipeline
├── constraint.py             # Fairness constraints (updated for ATE fix & joint Minimax loss)
├── load_data_utilities.py    # Preprocessing utilities (updated for dual sex/race extraction)
├── evaluate_intersection.py  # Standalone intersectional bias evaluation script
├── FairTrade.py              # Main training/BO orchestration pipeline
├── FairTrade-crypten.py      # Secure MPC training pipeline
│
├── requirements.txt          # Replicated python virtual environment freeze
└── README.md                 # Project documentation & execution guide
```

## Environment Setup & Prerequisites
All scripts were verified using Python 3.13. To configure your environment and guarantee exact metric reproducibility:

# Create and activate your virtual environment
```bash
python3 -m venv fair_env
source fair_env/bin/activate

# Install pinned system dependencies
pip install -r requirements.txt

# Create the results directory to store generated .npy files before executing the code
mkdir -p results/adult
```

## Execution Commands
To ensure total experimental reproducibility across client allocations, running any script automatically initializes a deterministic random seed (seed=45).

### Task 1: Replicating Baseline Demographic Parity (Adult R3C)
To execute the baseline training pipeline for 50 communication rounds over 3 clients with random partitioning:
```bash
python3 FairTrade_pre_changes.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'adult' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type 'random'
```
Note: The FairTrade.py was modified for the tasks as part of the challenge - The FairTrade file in the repo will execute Task 3 out of the box. if you wish to refer the original FairTrade file, please refer FairTrade_pre_changes.py
### Task 2: Executing the Intersectional Evaluation
To audit single-attribute vs. intersectional demographic biases:
```bash
python3 evaluate_intersection.py
```
### Task 3: Joint Fairness Optimization Run (Minimax Max-SPD)
To run our joint-attribute optimization model (optimizing gender and race parity metrics concurrently):
```bash
python3 FairTrade.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'adult' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type 'random'
```
## References
If you find this work useful in your research, please consider citing:
```bash
@inproceedings{badar2024fairtrade,
  title={FairTrade: Achieving Pareto-Optimal Trade-offs Between Balanced Accuracy and Fairness in Federated Learning},
  author={Badar, Maryam and Sikdar, Sandipan and Nejdl, Wolfgang and Fisichella, Marco},
  booktitle={Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence},
  year={2024}
}
@inproceedings{agarwal2018,
  title={A reductions approach to fair classification},
  author={Agarwal, Alekh and Beygelzimer, Alina and Dud{\'\i}k, Miroslav and Langford, John and Wallach, Hanna},
  booktitle={International Conference on Machine Learning},
  pages={60--69},
  year={2018},
  organization={PMLR}
}
@inproceedings{verma2018,
  title={Fairness definitions explained},
  author={Verma, Sahil and Rubin, Julia},
  booktitle={2018 IEEE/ACM international workshop on software fairness (FairWare)},
  pages={1--7},
  year={2018},
  organization={IEEE}
}
```
### AI usage acknowledgment:
Google Gemini (2026) assisted in code analysis, debugging framework runtime crashes, optimization loss formulation verification, and documentation structure optimization.
