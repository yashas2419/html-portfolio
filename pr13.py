#naive bayes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score

# Load and preprocess the dataset
data = pd.read_csv('pr5.csv')  # Replace with your CSV file path
X, y = data.iloc[:, :-1], data.iloc[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Naive Bayes classifier and evaluate
model = GaussianNB().fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(precision_score(y_test,model.predict(X_test)))
print(recall_score(y_test,model.predict(X_test)))
print(confusion_matrix(y_test,model.predict(X_test)))
print(f"Accuracy of the Naive Bayes Classifier: " , accuracy)