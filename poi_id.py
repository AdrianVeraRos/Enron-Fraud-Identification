#!/usr/bin/python

import sys
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import sklearn
import pickle

print('The scikit-learn version is {}.'.format(sklearn.__version__))


sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import tester
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
feat_1 = [
    "poi",
    "salary",
    "deferral_payments",
    "total_payments",
    "loan_advances",
    "bonus",
    "restricted_stock_deferred",
    "deferred_income",
    "total_stock_value",
    "expenses",
    "exercised_stock_options",
    "other",
    "long_term_incentive",
    "restricted_stock",
    "director_fees",
    "to_messages",
    "from_poi_to_this_person",
    "from_messages",
    "from_this_person_to_poi",
    "shared_receipt_with_poi"
    ]
features_list = feat_1

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
data_dict.pop("TOTAL")
data_dict.pop("LOCKHART EUGENE E")

def nan_replacer(dataset): # Sets NaN financial features to 0
    for f in features_list:
        for person in dataset:
            if dataset[person][f] == "NaN":
                dataset[person][f] = 0
    return dataset

data_dict = nan_replacer(data_dict)

### Task 3: Create new feature(s)
for person in data_dict:
    dp = data_dict[person]
    from_poi = dp['from_poi_to_this_person']
    to_poi = dp['from_this_person_to_poi']
    to_m = dp['to_messages']
    from_m = dp['from_messages']
    t_stock = dp['total_stock_value']
    t_income = dp['total_payments']+dp['total_stock_value']

    if to_m != 0:
        dp['from_poi_ratio'] = float(from_poi) / float(to_m)
    else:
        dp['from_poi_ratio'] = 0
    if from_m != 0:
        dp['to_poi_ratio'] = float(to_poi) / float(from_m)
    else:
        dp['to_poi_ratio'] = 0
    if t_stock != 0:
        dp['stock_ratio'] = float(t_stock) / float(t_income)
    else:
        dp['stock_ratio'] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Select K Best initial
'''
selector = SelectKBest(f_classif, k='all')
selector.fit(features, labels)
scores = zip(features_list[1:], selector.scores_)
sorted_scores = sorted(scores)
print 'SKBest scores: ', sorted_scores
'''

feat_top15 = ['poi',
              'total_payments',
              'total_stock_value',
              'from_poi_to_this_person',
              'shared_receipt_with_poi',
              'expenses',
              'deferred_income',
              'long_term_incentive',
              'other',
              'bonus',
              'restricted_stock',
              'salary',
              'exercised_stock_options',
              'to_poi_ratio', # new feature created
              'from_poi_ratio',# new feature created
              'stock_ratio'] # new feature created

features_list = feat_top15
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#Due to having overfit & 0 recall/precision problems with the provided splitter for training,
#we will use the tester.py StratifiedShuffleSplit splitter.

from sklearn.cross_validation import StratifiedShuffleSplit
folds = 1000
cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
for train_idx, test_idx in cv:
    features_train = []
    features_test = []
    labels_train = []
    labels_test = []
    for ii in train_idx:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
    for jj in test_idx:
        features_test.append(features[jj])
        labels_test.append(labels[jj])

## Initial algorithms scores
clf_AB = AdaBoostClassifier()
tester.test_classifier(clf_AB, data_dict, features_list)
clf_RBF = SVC(kernel='rbf', max_iter=1000)
tester.test_classifier(clf_RBF, data_dict, features_list)
clf_RF = RandomForestClassifier()
tester.test_classifier(clf_RF, data_dict, features_list)
clf_SVC = SVC(kernel='linear', max_iter=1000)
tester.test_classifier(clf_SVC, data_dict, features_list)
clf_NB = GaussianNB()
tester.test_classifier(clf_NB,data_dict,features_list)
clf_KNN = KNeighborsClassifier()
tester.test_classifier(clf_KNN,data_dict,features_list)


# ### Task 5: Tune your classifier to achieve better than .3 precision and recall
# ### using our testing script. Check the tester.py script in the final project
# ### folder for details on the evaluation method, especially the test_classifier
# ### function. Because of the small size of the dataset, the script uses
# ### stratified shuffle split cross validation. For more info:
# ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Adaboost tuneup
scaler = MinMaxScaler()
skb = SelectKBest()

clf = AdaBoostClassifier()
pipe = Pipeline([('Scale_Features', scaler), ('SKB', skb), ('Classifier', clf)])

print sorted(pipe.get_params().keys())

params = {'SKB__k': range(1, len(features_list)),
          'Classifier__n_estimators': [1, 5, 10, 25, 50]}
my_clf = GridSearchCV(pipe, param_grid=params, scoring='f1')
my_clf.fit(features_train, labels_train)

print "Adaboost"
pred = my_clf.predict(features_test)
print("Best estimator found by grid search:")
print my_clf.best_estimator_
print('Best Params found by grid search:')
print my_clf.best_params_


ada = AdaBoostClassifier(n_estimators = 10)
selection = SelectKBest(k=13)
pipeline = Pipeline([("scale", scaler), ("features", selection), ("classifier", ada)])
clf = pipeline
tester.test_classifier(clf,data_dict,features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


