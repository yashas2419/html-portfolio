import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
df = pd.read_csv("pr5.csv")
col_names = ['num_preg', 'glucouse', 'bp', 'thickness', 'insulin', 'bmi', 'dia_pred', 'age']
class_names = ['diabetes']
X = df[col_names].values 
y = df[class_names].values

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33)
print ('\n the total number of Training Data :',ytrain.shape)
print ('\n the total number of Test Data :',ytest.shape)
clf = GaussianNB()
clf.fit(xtrain,ytrain.ravel())
y_pred = clf.predict(xtest)
predictTestData= clf.predict([[6,178,92,35,0,33.6,0.627,80]])
print('\n Confusion matrix')
print(metrics.confusion_matrix(ytest,y_pred))
print('\n Accuracy of the classifier is',metrics.accuracy_score(ytest,y_pred))
print('\n The value of Precision', metrics.precision_score(ytest,y_pred))
print('\n The value of Recall', metrics.recall_score(ytest,y_pred))
print("Predicted Value for individual Test Data:", predictTestData)