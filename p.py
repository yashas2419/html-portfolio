import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# New sample
new_sample = np.array([[2.1, 0.2, 3.4, 0.2]])

# Get the decision path
node_indicator = clf.decision_path(new_sample)
leaf_id = clf.apply(new_sample)

# Print the decision path
for sample_id in range(len(new_sample)):
    node_index = node_indicator.indices[node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]]
    
    print(f"Sample {sample_id}:")
    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            print(f"Leaf node {node_id}: Predicted class '{iris.target_names[clf.classes_[np.argmax(clf.tree_.value[node_id])]]}'")
            continue
        
        if new_sample[sample_id, clf.tree_.feature[node_id]] <= clf.tree_.threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        
        print(f"Node {node_id}: (Feature '{iris.feature_names[clf.tree_.feature[node_id]]}' = {new_sample[sample_id, clf.tree_.feature[node_id]]}) "
              f"{threshold_sign} {clf.tree_.threshold[node_id]}")
