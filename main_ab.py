"""
main_ab.py 

(AdaBoostClassifier)

CS 451 (Machine Learning) Final Project

Created by Daniel Trauner and Xi Wang on 2013-11-21.
Copyright (c) 2013 Daniel Trauner and Xi Wang. 
All rights reserved.

WORK LOG:
---------
11/21/13 -- 3:30PM -> 5:30PM -- Started work
11/23/13 -- 3:30PM -> 6:30PM -- Cleaned up hackish code
11/25/13 -- 4:00PM -> 6:00PM -- Performed experimentation
"""

import csv				
from random import shuffle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
import numpy

dictionary = {}

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def build_global_dict(parsed_data):
	'''
	Adds all of the word features in parsed_data
	to the global dictionary
	'''
	global dictionary
	
	hash_num = max(dictionary.values())+1 if dictionary else 0
	
	for person in parsed_data:
		for i in range(6,len(person)):
			if person[i] not in dictionary:
				dictionary[person[i]] = hash_num
				hash_num += 1

def parse_csv(path_to_csv_file):
	'''
	Returns an array containing a cleaned up version
	of the raw CSV data in the CSV file located at
	the given path.
	'''
	parsed_data = []

	with open(path_to_csv_file, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',', quotechar='\"')

		header = reader.next()[1:]

		for row in reader:
			parsed_data.append(row[1:])

	build_global_dict(parsed_data)

	return parsed_data

def encode_data(parsed_data):
	global dictionary
	encoded_data = []
	for person in parsed_data:
		applicant = []
		for i, feature in enumerate(person):
			if i < 6:
				applicant.append(int(feature))
			else:
				applicant.append(dictionary[feature])
		encoded_data.append(applicant)

	return encoded_data

def train_test(preprocessed_data):
	'''
	Trains a AdaBoostClassifier (using decision stumps)
	from sklearn using encoded_data where encoded_data 
	is a list of lists where each list is an example.
	'''
	all_labels = list(person[0] for person in preprocessed_data)
	all_features = list(person[4:] for person in preprocessed_data)

	print '\nPerforming a 10-fold cross validation with', len(preprocessed_data), 'examples...\n'
	ab_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=49, learning_rate=0.1681744)
	scores = cross_val_score(ab_classifier, numpy.array(all_features), numpy.array(all_labels), cv=10)

	print 'AdaBoostClassifier with:' 
	print '\tbase_estimator=' + str(ab_classifier.base_estimator)
	print '\tn_estimators=' + str(ab_classifier.n_estimators)
	print '\tlearning_rate=' + str(ab_classifier.learning_rate)
	print '\nAccuracy:', scores.mean(), '+/-', scores.std(), '\n'

def main():
	parsed_training_data = parse_csv('data.csv')
	encoded_training_data = encode_data(parsed_training_data)
	train_test(encoded_training_data)

if __name__ == '__main__':
	main()
