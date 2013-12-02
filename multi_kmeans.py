"""
multi_kmeans.py 

(KMeans)

CS 451 (Machine Learning) Final Project

Created by Daniel Trauner and Xi Wang on 2013-11-21.
Copyright (c) 2013 Daniel Trauner and Xi Wang. 
All rights reserved.

WORK LOG:
---------
12/2/13 -- 4:30PM -> ?:??PM -- Started work
"""

import csv
from random import shuffle
from sklearn.cluster import KMeans
import numpy

dictionary = {}

def frange(x, y, inc):
	'''
	A generator that acts as a range() for float
	values.
	'''
	while x < y:
		yield x
		x += inc

def build_global_dict(parsed_data):
	'''
	Adds all of the word features in parsed_data
	to the global dictionary
	'''
	global dictionary
	
	hash_num = max(dictionary.values())+1 if dictionary else 0
	
	for person in parsed_data:
		for i in range(3,len(person)):
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

		header = reader.next()

		for row in reader:
			parsed_data.append(row)

	build_global_dict(parsed_data)

	return parsed_data

def encode_data(parsed_data):
	global dictionary
	encoded_data = []
	for person in parsed_data:
		applicant = []
		for i, feature in enumerate(person):
			if i < 3:
				applicant.append(int(feature))
			else:
				applicant.append(dictionary[feature])
		encoded_data.append(applicant)

	return encoded_data

def train_test(preprocessed_data):
	'''
	Trains a KMeans cluster classifier from sklearn 
	using encoded_data where encoded_data  is a list 
	of lists where each list is an example.  Note that
	there are really no parameters to tweak.
	'''
	all_labels = list(person[0] for person in preprocessed_data)
	all_features = list(person[1:] for person in preprocessed_data)

	n = 100

	km_classifier = KMeans(n_clusters=5, n_init=n, max_iter=300, tol=0.0001, precompute_distances=True, n_jobs=2)
	predictions = km_classifier.fit_predict(preprocessed_data[1:]).tolist()

	print all_labels[0:5]
	print predictions[0:5]

def main():
	parsed_training_data = parse_csv('multi_data.csv')
	encoded_training_data = encode_data(parsed_training_data)
	train_test(encoded_training_data)

if __name__ == '__main__':
	main()