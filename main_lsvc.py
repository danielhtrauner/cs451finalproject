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
11/30/13 -- 6:00PM -> 9:00PM -- Performed experimentation

"""

import csv
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
import numpy
import string

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

def train_test(preprocessed_data):
	'''
	Trains a LinearSVC from sklearn using encoded_data
	where encoded_data is a list of lists where each 
	list is an example.
	'''
	all_labels = list(person[0] for person in preprocessed_data)
	all_features = list(person[4:] for person in preprocessed_data)

	print '\nPerforming a 10-fold cross validation with', len(preprocessed_data), 'examples...\n'
	lsvc_classifier = LinearSVC(penalty='l2', loss='l2', C=1.167, dual = False, tol = 1e-15)
	lsvc_accuracies = cross_val_score(lsvc_classifier, numpy.array(all_features), numpy.array(all_labels), cv=10)

	print 'LinearSVC with:' 
	print '\tpenalty=' + str(lsvc_classifier.penalty)
	print '\tloss=' + str(lsvc_classifier.loss)
	print '\tC=' + str(lsvc_classifier.C)
	print '\nAccuracy:', lsvc_accuracies.mean(), '+/-', lsvc_accuracies.std(), '\n'

def train_test_para(preprocessed_data, penalty, loss, c):
	'''
	This is the helper function for testing the best parameters.
	'''
	all_labels = list(person[0] for person in preprocessed_data)
	all_features = list(person[4:] for person in preprocessed_data)

	p = str(penalty)
	l = str(loss)

	lsvc_classifier = LinearSVC(penalty=p, loss=l, C=c, dual = False, tol = 1e-15)
	lsvc_accuracies = cross_val_score(lsvc_classifier, numpy.array(all_features), numpy.array(all_labels), cv=10)

	return lsvc_accuracies

def run(data):

	b_c = 0
	b_penalty = 'l1'
	b_loss = 'l2'
	b_acc = train_test_para(data, 'l1', 'l2', 1)

	# l1 penalty and l2 loss
	for c in numpy.arange(0.1, 1.5, 0.001):
		acc = train_test_para(data, 'l1', 'l2', c)
		if acc.mean() > b_acc.mean():
			b_acc = acc
			b_c = c
			b_loss = 'l2'
			b_penalty = 'l1'
	print "Done with l1 penalty and l2 loss"

	# l2 penalty and l2 loss
	for c in numpy.arange(0.1, 1.5, 0.001):
		acc = train_test_para(data, 'l2', 'l2', c)
		if acc.mean() > b_acc.mean():
			b_acc = acc
			b_c = c
			b_loss = 'l2'
			b_penalty = 'l2'

	print 'LinearSVC with:' 
	print '\tpenalty=' + str(b_penalty)
	print '\tloss=' + str(b_loss)
	print '\tC=' + str(b_c)
	print '\nAccuracy:', b_acc.mean(), '+/-', b_acc.std(), '\n'

def main():
	parsed_training_data = parse_csv('data.csv')
	encoded_training_data = encode_data(parsed_training_data)
	train_test(encoded_training_data)
	#run(encoded_training_data)

if __name__ == '__main__':
	main()