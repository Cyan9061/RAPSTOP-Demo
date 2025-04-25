------

# 🚀 RAP-STOP: Reliable LLM-Powered Multimodal Semantic Top-𝑘 Operator (DEMO)

Modern data lakes hold vast collections of multimodal data, and retrieving the top-𝑘 most relevant items across both structured and unstructured formats is crucial yet difficult. Traditional weighted-sum approaches lack true semantic understanding, and ML-based rankers require extensive labeled data—human experts remain the gold standard, but are expensive and slow.

**RAP-STOP** (Reliable LAnguage model Powered Semantic Top-𝑘 OPerator) bridges this gap by harnessing LLMs 🤖 for semantic ranking while mitigating their hallucination issues to deliver **reliable**, **efficient**, and **robust** top-𝑘 retrieval.

This repository provides a demo implementation of the associated research paper, demonstrating the recognition and ranking of individuals’ ages based on their images. 📊

You can also apply this approach to any other top-k or ranking task you wish! 🎉

------

## 🏆 Why RAP-STOP?

- **🔍 Hallucination Debugging**: Employs Monte Carlo Tree Search (MCTS) to identify and correct structural inconsistencies before ranking.
- **🎯 Credibility-Aware Ranking**: A learnable attention mechanism dynamically weights evidence, reducing hallucinated noise.
- **⚡ Enhanced Accuracy**: Achieves over 100% improvement compared to non-learning baselines (PageRank, nDegree, Copeland).
- **⏱️ Cost & Time Efficiency**: Delivers ≥90% human-level accuracy at under 0.1% of the time and cost of crowd-sourced labeling.
- **💰 Budget-Friendly LLMs**: Supports free or low-cost models (<$0.14 per million tokens), with seamless integration for premium models.


------

## 🗂️ Project Structure

```
.
├── data/
│   ├── sorting_target/           # 📸 Original topk target images (ages: 20, 27, 69, 85)
│   ├── evidenceMatrix.npy        # 🗳️ Original evidence matrix
│   ├── evidenceMatrix_debug.npy  # 🛠️ Matrix debugged via MCTS
│   └── realOrder.npy             # ✔️ Ground-truth ordering matrix
├── E2E_topk_experiment/          # ⚙️ End-to-end training and evaluation
│   ├── train_topk_operator.py    # 🏋️ Trains the AttentionSorter model
│   └── evaluate_topk_operator.py # 📊 Evaluates ranking accuracy
├── MCTS/                         # 🌲 Monte Carlo Tree Search utilities
│   ├── mcts/                     # 🧩 Core MCTS modules
│   └── mcts_debug.py             # 🔧 Debugs evidenceMatrix.npy
├── models/                       # 🧠 Saved AttentionSorter checkpoints
└── README.md                     # 📖 This file
```

------

## 📦 Requirements

- **Python**: 3.10.16
- **NumPy**: 2.0.1
- **PyTorch**: 2.4.0
- **CUDA Toolkit**: ≥11.8


------

## 📥 Setup & Data Preparation

1. **Prepare Sorting Targets** 🤳
   - Four images (ages: 20, 27, 69, 85) have been placed into `data/sorting_target/`.
   - We have used LLMs to generate and save `data/evidenceMatrix.npy`.

2. **Provide Ground-Truth Order** ✅
   - Store the true top-𝑘 order in `data/realOrder.npy` for evaluation.

------

## 🧪 Usage Examples

### 1. Train the Attention Top‑𝑘 Operator
```bash
python .\E2E_topkExperiment\train_topk_operator.py --lr 1e-4 --bs 32 --trainEpochs 100
```
Model checkpoints will be saved in `models/`.

### 2. Debug Evidence Matrix with MCTS
```bash
python .\MCTS\mcts_debug.py --budgetP 0.01 --explorationConstant 1 --epsilon 1e-5 --evidenceLoadPath evidenceMatrix.npy --savePath evidenceMatrix_debug.npy
```
This step resolves hallucination conflicts and outputs `evidenceMatrix_debug.npy`.

### 3. Evaluate Ranking Performance

- **Original Evidence Matrix:**
    ```bash
    python .\E2E_topkExperiment\evaluate_topk_operator.py --evidenceMatrix evidenceMatrix.npy --realOrder realOrder.npy --lr 1e-4 --bs 32 --model_path Visual_K9_M16_N5000_bs32_lr0.0001.pt --result_path evalResult.csv
    ```

- **Debugged Evidence Matrix:**
    ```bash
    python .\E2E_topkExperiment\evaluate_topk_operator.py --evidenceMatrix evidenceMatrix_debug.npy --realOrder realOrder.npy --lr 1e-4 --bs 32 --model_path Visual_K9_M16_N5000_bs32_lr0.0001.pt --result_path evalResult_debug.csv
    ```

------

## 📈 Results & Impact

Using **RAP-STOP**, we demonstrate:
- **🔥 Over 100% accuracy improvement** compared to non-learning baselines.
- **💪 ≥90% human-level performance** at <0.1% of time and cost.
- **⚙️ Seamless integration with existing LLMs** at minimal expense.

------

### 🧙 What it does:

- 🕵️ **Hallucination Error Detection**: Identifies and corrects structural inconsistencies in LLM-generated evidence through Monte Carlo Tree Search (MCTS)
- ⚖️ **Dynamic Evidence Weighting**: Utilizes a learnable attention mechanism to dynamically weight evidence, minimizing the impact of hallucinated noise
- 🌐 **Cross-Modal Ranking**: Handles multimodal data (text/image/numerical and more) through isomorphic partial order representation
- 🚀 **Massive Parallel Comparisons**: Executes up to 10^4 pairwise LLM queries per minute via concurrent API calls
- 🧠 **Zero-Shot Transfer Learning**: Trains on synthetic data then transfers ranking capability to unseen domains


------
Ready to supercharge your multimodal top-κ retrieval? Let **RAP-STOP** handle the heavy lifting! ⚡🔍
