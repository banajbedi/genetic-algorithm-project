import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tpot.builtins import ZeroCount
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('data.csv')
tpot_data['Class'] = tpot_data['Class'].map({'g':0, 'h':1})
features = tpot_data.drop('Class', axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(features, tpot_data['Class'], random_state=42)

# Average CV score on the training set was: 0.8564318261479145
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=84),
    ZeroCount(),
    StandardScaler(),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=1, max_features=0.35000000000000003,
                               min_samples_leaf=4, min_samples_split=7, n_estimators=100, subsample=0.55)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print(accuracy_score(testing_target, results))