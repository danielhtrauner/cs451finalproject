"""
main_knn.py 

(KNeighborsClassifier)

CS 451 (Machine Learning) Final Project

Created by Daniel Trauner and Xi Wang on 2013-11-21.
Copyright (c) 2013 Daniel Trauner and Xi Wang. 
All rights reserved.

WORK LOG:
---------
11/21/13 -- 3:30PM -> 5:30PM -- Started work
11/23/13 -- 3:30PM -> 6:30PM -- Cleaned up hackish code
11/29/13 -- 6:30PM -> 7:00PM -- Performed experimentation
12/01/13 -- 2:30PM -> ?:??PM -- Added back testing code
"""

import csv				
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import numpy

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
	Trains a KNeighborsClassifier from sklearn
	using encoded_data where encoded_data is a list
	of lists where each list is an example.
	'''
	avg_acc = 0.0
	n = 100
	for i in range(n):
		shuffle(preprocessed_data)

		all_labels = list(person[0] for person in preprocessed_data)
		all_features = list(person[4:] for person in preprocessed_data)

		knn_classifier = KNeighborsClassifier(n_neighbors=73, weights='distance')
		scores = cross_val_score(knn_classifier, numpy.array(all_features), numpy.array(all_labels), cv=10)

		avg_acc += scores.mean()

	print '\nThe average accuracy of', n, '10-fold CVs is', avg_acc/n, 'for an optimized KNeighborsClassifier.\n'

	# TESTING CODE
	# ------------
	# best_k = 0
	# best_w = ''
	# best_acc = 0
	# best_std = 0
	# i = 1
	# for k_i in range(1,len(preprocessed_data)):
	# 	for w_i in ['uniform', 'distance']:
	# 		# print '\nPerforming a 10-fold cross validation with', len(preprocessed_data), 'examples...\n'
	# 		knn_classifier = KNeighborsClassifier(n_neighbors=k_i, weights=w_i)
	# 		scores = cross_val_score(knn_classifier, numpy.array(all_features), numpy.array(all_labels), cv=10)

	# 		# print 'KNeighborsClassifier with:' 
	# 		# print '\tn_neighbors=' + str(knn_classifier.n_neighbors)
	# 		# print '\tweights=' + str(knn_classifier.weights)
	# 		# print '\nAccuracy:', scores.mean(), '+/-', scores.std(), '\n'

	# 		if scores.mean() > best_acc:
	# 			best_k = k_i
	# 			best_w = w_i
	# 			best_acc = scores.mean()
	# 			best_std = scores.std()
	# 		print chr(27) + "[2J"
	# 		print 'In', i, 'iterations the KNeighborsClassifier with the current highest accuracy has:' 
	# 		print '\tn_neighbors=' + str(best_k)
	# 		print '\tweights=' + str(best_w)
	# 		print '\nAccuracy:', best_acc, '+/-', best_std, '\n'

	# 		i += 1

	# print chr(27) + "[2J"
	# print '='*100
	# print 'The KNeighborsClassifier with the highest overall accuracy had:' 
	# print '\tn_neighbors=' + str(best_k)
	# print '\tweights=' + str(best_w)
	# print '\nAccuracy:', best_acc, '+/-', best_std, '\n'

def main():
	parsed_training_data = parse_csv('binary_data.csv')
	encoded_training_data = encode_data(parsed_training_data)
	train_test(encoded_training_data)

if __name__ == '__main__':
	main()