#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
s = sorted([(k[0],k[1]) for k in data])
h1,h2,h3,h4 = s[-1][1], s[-2][1], s[-3][1], s[-4][1]
for a in data_dict:
    h = data_dict[a]['bonus']
    if h == h1 or h == h2 or h == h3 or h == h4:
        print(a, data_dict[a]['salary'], data_dict[a]['bonus'])


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

