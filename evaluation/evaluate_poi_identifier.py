#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from time import time
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

print(len(features_test))
from sklearn import tree
clf = tree.DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train) 
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
test1= [0.]*len(pred)
print "pred time:", round(time()-t1, 3), "s"
print(len(labels_test))
print(len([predicted for predicted, label in zip(pred, features_test) if predicted == label]))
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print(recall_score(true_labels, predictions))

print(precision_score(features_test, pred))
print(recall_score(features_test, pred))
 
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
acc1 = accuracy_score(test1, labels_test)
print "accuracy ", acc, "s"
print "identifier predicted 0.0 ", acc1, "s"



