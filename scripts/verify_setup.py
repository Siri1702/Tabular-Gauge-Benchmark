# scripts/verify_setup.py
import os
token = 'tabpfn_sk_88f0WVLj1xZ2C92jTr5MtU31ms5bmVOdCsC6eABflnw'
import tabpfn_client
tabpfn_client.set_access_token(token)
from tabpfn_client import TabPFNClassifier
import inspect
print(inspect.signature(TabPFNClassifier))
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = TabPFNClassifier()
print(clf)  # or 'cuda' if available

t0 = time.time()
clf.fit(X_train, y_train)
train_time = time.time() - t0

t0 = time.time()
proba = clf.predict_proba(X_test)[:, 1]
infer_time = time.time() - t0

auc = roc_auc_score(y_test, proba)
print(f"TabPFN AUC: {auc:.4f}")
print(f"Fit time: {train_time:.3f}s | Inference time: {infer_time:.3f}s")
# Expected: AUC ~0.995, fit ~2s, inference ~0.5s on CPU