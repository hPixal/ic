import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score

# Cargar el conjunto de datos Wine
wine = load_wine()
X = wine.data
y = wine.target

# Crear clasificadores base
base_classifier = DecisionTreeClassifier(max_depth=1)

# Crear y configurar el clasificador Bagging
bagging = BaggingClassifier(estimator=base_classifier, 
                             n_estimators=50, 
                             random_state=42)

# Crear y configurar el clasificador AdaBoost
adaboost = AdaBoostClassifier(estimator=base_classifier, 
                               n_estimators=50, 
                               algorithm='SAMME.R', 
                               random_state=42)

# Evaluar el rendimiento usando validación cruzada de 5 particiones
cv_folds = 5
bagging_scores = cross_val_score(bagging, X, y, cv=cv_folds, scoring='accuracy')
adaboost_scores = cross_val_score(adaboost, X, y, cv=cv_folds, scoring='accuracy')

# Imprimir resultados
print("Resultados de Bagging:")
print("Precisión media:", np.mean(bagging_scores))
print("Desviación estándar de precisión:", np.std(bagging_scores))

print("\nResultados de AdaBoost:")
print("Precisión media:", np.mean(adaboost_scores))
print("Desviación estándar de precisión:", np.std(adaboost_scores))
