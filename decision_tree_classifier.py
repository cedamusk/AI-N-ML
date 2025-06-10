import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

# 1. Generate a synthetic binary classification dataset
X, y=make_classification(
    n_samples=200,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

#Convert o a DataFrame for clarity
feature_names=['Feature A', 'Feature B', "Feature C", 'Feature D']
df=pd.DataFrame(X, columns=feature_names)
df['Label']=y

# 2. Split into training and test sets
X_train, X_test, y_train, y_test=train_test_split(
    df[feature_names], df["Label"], test_size=0.2, random_state=42
)

#3. Train a decision tree classifies
clf=DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)

clf.fit(X_train, y_train)

#4. Evaluate using cross-validation (simulates pruning phase)
cv_scores=cross_val_score(clf, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")

#5. Test set performance
y_pred=clf.predict(X_test)
print("\nTest Set Classification Report:\n")
print(classification_report(y_test, y_pred))

#6. Visualize the decision tree
plt.figure(figsize=(12, 6))
plot_tree(
    clf,
    filled=True,
    feature_names=feature_names,
    class_names=["Class 0", "Class 1"],
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Classifier (Synthetic Data)")
plt.tight_layout()
plt.savefig("decision_tree_plot.png", dpi=300)
