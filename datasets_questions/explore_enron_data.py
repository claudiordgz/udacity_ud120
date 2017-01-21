#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print(len(enron_data))

def stuff(data, key):
    a = data[key]
    a['name'] = key
    return a


pois = [stuff(enron_data, p) for p in enron_data.keys() if enron_data[p]['poi'] == 1]
print(len(pois))
sorted_pois = sorted(pois, key=lambda x: x["total_payments"], reverse=True)
print(["{name} - {money}".format(name=p['name'], money=p["total_payments"]) for p in sorted_pois])

salaried_people = [stuff(enron_data, p) for p in enron_data.keys()]
print(len(["{name} - {money}".format(name=p['name'], money=p["total_payments"]) for p in salaried_people if p["total_payments"] == 'NaN']))
