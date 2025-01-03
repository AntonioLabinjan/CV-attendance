BOILERPLATE/TEMPLATE ZA ROC KRIVULJU => ŽELIMO BIT ČA BLIŽE GORNJEMU LIVEMU KUTU; ONA CRTA PO DIJAGONALI PREDSTAVLJA RANDOM 50/50 CLASSIFIER

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Podaci u formatu: k1, k2, threshold, accuracy, recall, f1, tp, fp, tn, fn
data = [
    (5, 10, 0.5, 0.6910, 0.6910, 0.6632, 2556, 774, 36378, 774),
    (5, 10, 0.55, 0.7121, 0.7121, 0.6660, 2634, 977, 45919, 977),
    (5, 10, 0.6, 0.7118, 0.7118, 0.6591, 2633, 1047, 49209, 1047),
    (5, 10, 0.7, 0.7078, 0.7078, 0.6542, 2618, 1077, 50619, 1077),
    (5, 10, 0.75, 0.7078, 0.7078, 0.6542, 2618, 1077, 50619, 1077),
    (5, 10, 0.8, 0.7078, 0.7078, 0.6542, 2618, 1077, 50619, 1077),
    (5, 10, 0.85, 0.7078, 0.7078, 0.6542, 2618, 1077, 50619, 1077),
    (5, 10, 0.9, 0.7078, 0.7078, 0.6542, 2618, 1077, 50619, 1077),
    (5, 10, 0.95, 0.7078, 0.7078, 0.6542, 2618, 1077, 50619, 1077),
    (5, 10, 1.0, 0.7078, 0.7078, 0.6542, 2618, 1077, 50619, 1077),
    # Dodaj ostale redke podataka prema potrebi...
]

# Listanje TPR i FPR
tpr = []
fpr = []

for _, _, threshold, _, _, _, tp, fp, tn, fn in data:
    # Izračunaj True Positive Rate (TPR) i False Positive Rate (FPR)
    tpr.append(tp / (tp + fn))  # TPR = TP / (TP + FN)
    fpr.append(fp / (fp + tn))  # FPR = FP / (FP + TN)

# Kreiraj ROC krivulju
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='o', label='ROC Curve')

# Random classifier linija (diagonalna linija)
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')

# Dodaj naslove, oznake i legendu
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Dodaj mrežu i prikazivanje grafa
plt.grid(True)
plt.show()
