# 🔬 Ablation Study: Label Shuffling

This directory contains the evidence for the **Label Shuffling Ablation Study**, which proves that the performance of Foundation Models in the few-shot regime is a result of genuine in-context learning of data patterns and not a result of dataset bias or random chance.

## 🧪 The Experiment
To test if the model was "cheating" or simply exploiting some property of the datasets, we performed the following:
1. **Control**: Evaluate the model on the original dataset.
2. **Ablation**: Randomly shuffle the labels of the training set while keeping the feature values intact. This destroys the relationship between $X$ and $y$.
3. **Comparison**: Compare the performance of the model on the shuffled data versus the original data.

## 📊 Results
- **File**: `shuffled_labels_results.csv`
- **Key Finding**: When labels are shuffled, the performance of Foundation Models drops to approximately $0.5$ (random guessing), while the l-advantage disappears. This confirms that the model is actively utilizing the training labels provided in the context to make predictions.

## 🏃 How to Replicate
To re-run this ablation study, execute:
```bash
python experiments/run_ablation_study.py
```
