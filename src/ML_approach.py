import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

directory='data/'

# Loading the data
df = pd.read_csv(directory + 'Train.csv')
sample_submission = pd.read_csv(directory + 'SampleSubmission.csv')

# DATA WRANGLING
# Keeping the square ids labels for submitting results.
square_ids = df[['Square_ID']]
# Creating a train and a test dataset
df.set_index('Square_ID', inplace=True)
columns_names = list(df.columns)

# Separating values for train (with target) and test sets.
colnames_train = ['target_2015','X', 'Y', 'LC_Type1_mode', 'elevation',
                  'precip 2014-11-16 - 2014-11-23', 'precip 2014-11-23 - 2014-11-30', 'precip 2014-11-30 - 2014-12-07',
                  'precip 2014-12-07 - 2014-12-14', 'precip 2014-12-14 - 2014-12-21', 'precip 2014-12-21 - 2014-12-28',
                  'precip 2014-12-28 - 2015-01-04', 'precip 2015-01-04 - 2015-01-11', 'precip 2015-01-11 - 2015-01-18',
                  'precip 2015-01-18 - 2015-01-25', 'precip 2015-01-25 - 2015-02-01', 'precip 2015-02-01 - 2015-02-08',
                  'precip 2015-02-08 - 2015-02-15', 'precip 2015-02-15 - 2015-02-22', 'precip 2015-02-22 - 2015-03-01',
                  'precip 2015-03-01 - 2015-03-08', 'precip 2015-03-08 - 2015-03-15']

colnames_test = ['X', 'Y', 'LC_Type1_mode', 'elevation',
                 'precip 2019-01-20 - 2019-01-27', 'precip 2019-01-27 - 2019-02-03', 'precip 2019-02-03 - 2019-02-10',
                 'precip 2019-02-10 - 2019-02-17', 'precip 2019-02-17 - 2019-02-24', 'precip 2019-02-24 - 2019-03-03',
                 'precip 2019-03-03 - 2019-03-10', 'precip 2019-03-10 - 2019-03-17', 'precip 2019-03-17 - 2019-03-24',
                 'precip 2019-03-24 - 2019-03-31', 'precip 2019-03-31 - 2019-04-07', 'precip 2019-04-07 - 2019-04-14',
                 'precip 2019-04-14 - 2019-04-21', 'precip 2019-04-21 - 2019-04-28', 'precip 2019-04-28 - 2019-05-05',
                 'precip 2019-05-05 - 2019-05-12', 'precip 2019-05-12 - 2019-05-19']

train = df.loc[:,colnames_train]
test = df.loc[:,colnames_test]

# Renaming precipitation columns to -17weeks until -1week (prior to the flooding).
train.columns = ['target_2015','X', 'Y', 'LC_Type1_mode', 'elevation',
                 '-17_weeks', '-16_weeks','-15_weeks', '-14_weeks', '-13_weeks', '-12_weeks',
                 '-11_weeks', '-10_weeks','-9_weeks', '-8_weeks', '-7_weeks', '-6_weeks',
                 '-5_weeks', '-4_weeks','-3_weeks', '-2_weeks', '-1_weeks']

test.columns = ['X', 'Y', 'LC_Type1_mode', 'elevation',
                '-17_weeks', '-16_weeks', '-15_weeks', '-14_weeks', '-13_weeks', '-12_weeks',
                '-11_weeks', '-10_weeks', '-9_weeks', '-8_weeks', '-7_weeks', '-6_weeks',
                '-5_weeks', '-4_weeks', '-3_weeks', '-2_weeks', '-1_weeks']

# SUM ALL THE PRECIPITATIONS IN ONE MORE FEATURE (TOTAL PRECIPITATIONS)

train['total_precipitations'] = train[['-17_weeks', '-16_weeks','-15_weeks', '-14_weeks', '-13_weeks', '-12_weeks',
                 '-11_weeks', '-10_weeks','-9_weeks', '-8_weeks', '-7_weeks', '-6_weeks',
                 '-5_weeks', '-4_weeks','-3_weeks', '-2_weeks', '-1_weeks']].sum(axis=1)

test['total_precipitations'] = test[['-17_weeks', '-16_weeks','-15_weeks', '-14_weeks', '-13_weeks', '-12_weeks',
                 '-11_weeks', '-10_weeks','-9_weeks', '-8_weeks', '-7_weeks', '-6_weeks',
                 '-5_weeks', '-4_weeks','-3_weeks', '-2_weeks', '-1_weeks']].sum(axis=1)


# ML MODEL
# TODO: For now we train on all the training dataset, we don't do a train-test split.

# APPROACH: First do a classification problem to get the places which are NOT flooded, to append directly at the end
# as 0's. With the flooded places, we will do a regression problem.
train_classification = train.drop(['X', 'Y', 'target_2015'], axis=1)
# Select the train_class_target, we place .copy() otherwise any modification modifies train['target_2015'] as well.
train_classification_target = train['target_2015'].copy()
# Replace the labels to a categorical problem.
train_classification_target[train_classification_target > 0] = 1
train['flooded_or_not'] = train_classification_target

# Binary classification problem (Random_forest):
RF = RandomForestClassifier(class_weight='balanced')
RF.fit(train_classification, train_classification_target.ravel())
pred_class = RF.predict(test.drop(['X', 'Y'], axis=1))
# TODO: THERE IS A PROBLEM WITH IMBALANCED CLASSES (ONLY 207 FLOODED AREAS APPEAR).

# Add the predictions to the test dataset as another variable for the regression algorithm
test['flooded_or_not'] = pred_class


# Create Pipeline
pipeline_xgb =Pipeline(
    [('xgb', xgb.XGBRegressor())
     ])

# Set the parameters
params={}
params['xgb__learning_rate'] = [0.05, 0.08]
params['xgb__objective'] = ['reg:squarederror']
params['xgb__max_depth'] = [3, 5]

# GridSearch
CV = GridSearchCV(pipeline_xgb, params, scoring = 'neg_mean_squared_error', n_jobs= 1, cv=3)
CV.fit(train.drop(['X', 'Y', 'target_2015'], axis=1), train['target_2015'])


# TEST SUBMISSION

predictions = CV.predict(test.drop(['X', 'Y'], axis=1))
predictions = pd.DataFrame(predictions)

# Remove the negative predictions (there shouldn't be any...) and put them to 0.
predictions[predictions < 0] = 0

# Redo the predictions dataframe into the right format to submit.
predictions.reset_index(inplace=True, drop=True)
submission = square_ids.join(predictions)
submission.columns = ['Square_ID', 'target_2019']

submission.to_csv('submission_ML.csv', index=False)


# TODO:ISSUES. Negative Predictions!