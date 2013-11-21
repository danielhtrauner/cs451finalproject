cs451finalproject
=================

CS 451 (Machine Learning) Final Project 
Daniel Trauner and Xi Wang 
Project Proposal 

---

Team: 

Daniel Trauner and Xi Wang

Summary:

For our final project, we plan to experiment with various machine learning algorithms in order to predict whether or not a prospective Middlebury student (based on their current state/region of residence, SAT verbal score, SAT math score, and stated major of interest) will be admitted to Middlebury.  Some of the variables we plan to experiment with include: which type of ML algorithm is used (supervised -- LinearSVC (SVM), KNN, and Random Forests -- or unsupervised -- KMeans), whether or not the problem is binary (predicting admitted vs. not admitted) or multi-class (predicting admitted reg enrolled, admitted reg not enrolled, admitted feb enrolled, admitted feb not enrolled, or not admitted) or some other multi-class variation, and whether or not it's best to use the more general versions of some of the features than it is to use their more specific counterparts (i.e. region vs. state of residence and major category vs. major).  It's likely we will focus on the problem of predicting enrollment as this could be especially valuable for the Admissions Office.  In order to evaluate the results of our experiments, we will be using training accuracy (using train-development-test splits) as well as various other metrics/tests to determine overfitting, and will generally attempt to fine-tune the parameters of each classifier using development data to obtain optimal parameter values before doing any cross-classifier comparison.
	
Resources:

(Data Set) Xi had the data from his ECON 211 class two years ago -- it describes students who applied to Middlebury in 2005 and had an on-campus interview.  The original data had around 1600 examples, but we whittled it down to 1391 examples in order to remove examples missing data for certain categories as well as to remove 33 examples which had admitted=0 but enrolled=1 (probably outliers where there were special circumstances involved resulting in their enrollment).  We also did minor feature engineering to extract more general versions of the state_name (census_region) and major (majorcat).  Each example (after our feature engineering) has nine features (including binary, real, and text features).  Note that we're in the process of obtaining newer admissions data from the Admissions Office.  In the meantime we'll use our older data, but we may switch data sets mid-project.
	
(Tools) We will be using Python and the scikit-learn library in order to run all of our experiments.
