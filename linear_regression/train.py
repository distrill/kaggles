import pandas as pd
import numpy as np
from sklearn import linear_model
# from sklearn import svm
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv('../train.csv')
test_data = pd.read_csv('../test.csv')

# separate X and y into their own df, encode categorical features
feature_labels = list(train_data.columns.values)[:-1]
dep_label = list(train_data.columns.values)[-1:]
X_df = pd.get_dummies(train_data.drop(dep_label, axis=1))
y_df = pd.get_dummies(train_data.drop(feature_labels, axis=1))

# sanitize data
X = X_df.values
y = y_df.values
X[np.isfinite(X) == False] = 0


# find alpha with highest cross-validation score
score = 0
for i in range(1, 10000):
  temp_model = linear_model.Ridge(alpha=i, fit_intercept=True)
  temp_model.fit(X,y)
  temp_score = np.mean(cross_val_score(temp_model, X, y, cv=5))
  if temp_score > score:
    alpha = i
    score = temp_score
    model = temp_model
  else:
    break

print("using alpha:", alpha)

# prepare test df (match the shape of training data)
test_X_df = pd.get_dummies(test_data).reindex(columns=X_df.columns, fill_value=0)
test_X = test_X_df.values
test_X[np.isfinite(test_X) == False] = 0

# see what we got
prediction = model.predict(test_X)
predicted_results = list(map(lambda x: x[0], prediction))

def formatForDf(index, result):
  return {'Id': index + len(X) + 1, 'SalePrice': result}

data = list(map(formatForDf, range(len(predicted_results)), predicted_results))
results_df = pd.DataFrame(data)

results_df.to_csv('ridge_results.csv', index=False)