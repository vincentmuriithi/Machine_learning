#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
import joblib
import numpy as np
import pandas as pd
red = 0
orange = 1
scars = 1
no_scar = 0
X = [[orange, 155, scars],[red, 123, no_scar],[red, 175, scars],
     [orange, 145,scars],[orange, 150, scars], [orange, 700, scars],
    [orange, 600, no_scar], [orange,60, scars], [red , 789, scars], 
    [orange, 800, scars], [orange, 200, scars]]
y = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]

model = DecisionTreeClassifier()
model.fit(X, y)
fruit = input("Enter the feautures of your fruit:\n").split()
fruit_features = [orange if fruit[0] == "orange" else red,
                  int(fruit[1]),
                      scars if fruit[2] == "scars" else no_scar]
fruit_features = np.array(fruit_features).reshape(1, -1)
pred = model.predict(fruit_features)
if pred == 1:
    print("it is an apple")
else:
    print("it is not an apple")
k = 0

plt.figure(figsize = (10,8))
plot_tree(model,feature_names=["Color", "Weight", "Scars"], class_names = ["Apple", "Not Apple"], filled = True)
plt.show()
joblib.dump(model,"apple{k}.pk1".f(k))