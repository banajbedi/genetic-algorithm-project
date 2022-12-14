import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import pickle

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('data.csv')
tpot_data['Class'] = tpot_data['Class'].map({'g':0, 'h':1})
features = tpot_data.drop('Class', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Class'], random_state=42)

# Average CV score on the training set was: 0.8552400981423063
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVC(C=10.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.001)),
    StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=0.01, fit_intercept=False, l1_ratio=1.0, learning_rate="constant", loss="log", penalty="elasticnet", power_t=100.0)),
    DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=13, min_samples_split=20)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

with open('pipeline_pkl', 'wb') as files:
    pickle.dump(exported_pipeline, files)
