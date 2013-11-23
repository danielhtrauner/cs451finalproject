"""
main_lsvc.py

(LinearSVC)

CS 451 (Machine Learning) Final Project

Created by Daniel Trauner and Xi Wang on 2013-11-21.
Copyright (c) 2013 Daniel Trauner and Xi Wang. 
All rights reserved.

WORK LOG:
---------
11/21/13 -- 3:30PM -> ?:??PM -- Started work
11/23/13 -- 3:30PM -> ?:??PM -- Cleaned up hackish code
"""

import csv
from random import shuffle
from sklearn.svm import LinearSVC

dictionary = {}

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

def split(data, fraction):
	'''
	Splits the data into two parts -- one that's the
	length of the given fraction of the entire set 
	and another comprising what's left over -- returning
	a tuple (maintaining the original order of the data).
	'''
	return (data[0:int(fraction*len(data))], data[int(fraction*len(data)):])

def train_test(preprocessed_data):
	'''
	Trains a LinearSVC from sklearn using encoded_data
	where encoded_data is a list of lists where each 
	list is an example.
	'''
	split_fraction = 0.9

	shuffle(preprocessed_data)

	# admitted = []

	# for person in preprocessed_data:
	# 	if person[0] == 1:
	# 		admitted.append(person)

	# preprocessed_data = admitted

	all_labels = split(list(person[0] for person in preprocessed_data), split_fraction)
	all_features = split((list(person[4:] for person in preprocessed_data)), split_fraction)

	# train on 90% of data
	train_labels = all_labels[0]
	train_features = all_features[0]

	# test on 10% of data
	test_labels = all_labels[1]
	test_features = all_features[1]

	print '\nTraining on', len(train_labels), 'examples...'
	lsvc_classifier = LinearSVC()
	lsvc_classifier.fit(train_features, train_labels)

	print 'Classifying', len(test_labels), 'examples...\n'
	lsvc_correct_count = 0.0
	for i in range(len(test_labels)):
		lsvc_predictions = lsvc_classifier.predict(test_features)
		lsvc_scores = lsvc_classifier.decision_function(test_features)
		if test_labels[i] == lsvc_predictions[i]:
			lsvc_correct_count += 1
		lsvc_accuracy = lsvc_classifier.score(test_features, test_labels)

	print 'LinearSVC:', lsvc_accuracy
	print 'The average distance to the hyperplane is', round(lsvc_scores.mean(), 2), '+/-', round(lsvc_scores.std(), 2), '\n'

def main():
	parsed_training_data = parse_csv('data.csv')
	encoded_training_data = encode_data(parsed_training_data)
	train_test(encoded_training_data)

if __name__ == '__main__':
	main()