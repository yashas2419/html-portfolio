'''import csv
a = []
with open('enjoysport.csv', 'r') as csvfile:
 for row in csv.reader(csvfile):
    a.append(row)
 print(a)
print("\n The total number of training instances are : ", len(a))
num_attribute = len(a[0]) - 1
print("\n The initial hypothesis is : ")
hypothesis = ['0'] * num_attribute
print(hypothesis)
for i in range(0, len(a)):
 if a[i][num_attribute] == 'yes':
    for j in range(0, num_attribute):
        if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
            hypothesis[j] = a[i][j]
        else:
            hypothesis[j] = '?'
 print("\n The hypothesis for the training instance {} is : \n".format(i + 1), hypothesis)
print("\n The Maximally specific hypothesis for the training instance is ")
print(hypothesis)



#2nd program


import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('enjoysport.csv')

# Extract concepts (features) and target values from the dataset
concepts = np.array(data.iloc[:, 0:-1])
print("Concepts:\n", concepts)
target = np.array(data.iloc[:, -1])
print("Target:\n", target)

def learn(concepts, target):
    # Initialize the specific hypothesis to the first positive instance
    specific_h = concepts[0].copy()
    print("Initialization of specific_h:")
    print(specific_h)

    # Initialize the general hypothesis with the most general hypothesis
    general_h = [["?" for i in range(len(specific_h))] for j in range(len(specific_h))]
    print("Initialization of general_h:")
    print(general_h)
    
    # Iterate over all the instances and update hypotheses
    for i, h in enumerate(concepts):
        print(f"Instance {i+1} - {h}")
        
        # If the target is positive, update the specific hypothesis
        if target[i] == "yes":
            print("Instance is Positive")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        
        # If the target is negative, update the general hypothesis
        if target[i] == "no":
            print("Instance is Negative")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        
        print(f"Step {i+1} - Specific_h: {specific_h}")
        print(f"Step {i+1} - General_h: {general_h}")
        print("\n")
    
    # Remove empty hypotheses from the general hypothesis
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    return specific_h, general_h

# Run the learning algorithm
s_final, g_final = learn(concepts, target)
print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")'''



#3rd program

import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) # two inputs [sleep,study]
y = np.array(([92], [86], [89]), dtype=float) # one output [Expected % in Exams]
X = X/np.amax(X,axis=0) # maximum of X array longitudinally
y = y/100

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = 2 #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons

wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons)) 
bh=np.random.uniform(size=(1,hiddenlayer_neurons)) 
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
    hinp1=np.dot(X,wh)
    hinp=hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1=np.dot(hlayer_act,wout)
    outinp= outinp1+ bout
    output = sigmoid(outinp)
    EO = y-output
    outgrad = derivatives_sigmoid(output)
    d_output = EO* outgrad
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad
# dotproduct of nextlayererror and currentlayerop
wout += hlayer_act.T.dot(d_output) *lr
wh += X.T.dot(d_hiddenlayer) *lr
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)