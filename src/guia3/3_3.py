import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import KFold

# Cargar el conjunto de datos Wine
wine = load_wine()
x = wine.data
y_d = wine.target

# Crear clasificadores base
dTree_1 = DecisionTreeClassifier(max_depth=1)
dTree_3 = DecisionTreeClassifier(max_depth=3)
dTree_5 = DecisionTreeClassifier(max_depth=5)

# Crear y configurar el clasificador Bagging
Bagging_1 = BaggingClassifier(estimator=dTree_1, 
                             n_estimators=50, 
                             random_state=42)

Bagging_3 = BaggingClassifier(estimator=dTree_3, 
                             n_estimators=50, 
                             random_state=42)

Bagging_5 = BaggingClassifier(estimator=dTree_5, 
                             n_estimators=50, 
                             random_state=42)

# Crear y configurar el clasificador AdaBoost
Adaboost_1 = AdaBoostClassifier(estimator=dTree_1, 
                               n_estimators=50,
                               algorithm="SAMME",
                               random_state=42)

Adaboost_3 = AdaBoostClassifier(estimator=dTree_3, 
                               n_estimators=50,
                               algorithm="SAMME", 
                               random_state=42)

Adaboost_5 = AdaBoostClassifier(estimator=dTree_5, 
                               n_estimators=50,
                               algorithm="SAMME", 
                               random_state=42)

classifiers = {
    'Bagginng Depth 1': Bagging_1,
    'Bagginng Depth 3': Bagging_3,
    'Bagginng Depth 5': Bagging_5,
    'AdaBoost Depth 1': Adaboost_1,
    'AdaBoost Depth 3': Adaboost_3,
    'AdaBoost Depth 5': Adaboost_5,
}

# Crear KFold con k particiones
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
ACCk = np.zeros(k)

print("-" * 40)
for name, clf in classifiers.items():
    print(f"Clasificador: {name}")

    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        # Crear conjuntos de entrenamiento y prueba para esta iteración
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_d[train_index], y_d[test_index]
        
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        ACCk[fold] = accuracy_score(y_test,y_pred)

    print(f"Media: {np.mean(ACCk)}")
    print(f"Varianza: {np.var(ACCk)}")
    print("-" * 40)
