import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv('../train.csv')
test_data = pd.read_csv('../test.csv')

# separate X and y into their own df, encode categorical features
feature_labels = list(train_data.columns.values)[:-1]
dep_label = list(train_data.columns.values)[-1:]
X_df = pd.get_dummies(train_data.drop(dep_label, axis=1))
y_df = pd.get_dummies(train_data.drop(feature_labels, axis=1))

print(X_df.shape)

# sanitize data
X = X_df.values
y = y_df.values
X[np.isfinite(X) == False] = 0

model = linear_model.Lasso(alpha=17, fit_intercept=True, tol=0.001)
model.fit(X, y)


# # see what we got
# prediction = model.predict(test_X)
# predicted_results = prediction

def getZeroedColumns(column):
  i = np.where(X_df.columns.values == column)
  if (model.coef_[i] == 0 and "Neighborhood_" not in column):
    return True
  return False

to_drop = list(filter(getZeroedColumns, X_df.columns.values))
X_df = X_df.drop(to_drop, axis=1)
print(len(X_df.columns.values))

# drop the columns that lasso had decided are butthole
X = X_df.values
X[np.isfinite(X) == False] = 0

model = linear_model.LinearRegression()
model.fit(X, y)

# # prepare test df (match the shape of training data)
test_X_df = pd.get_dummies(test_data).reindex(columns=X_df.columns, fill_value=0)
test_X = test_X_df.values
test_X[np.isfinite(test_X) == False] = 0
prediction = model.predict(test_X)
predicted_results = list(map(lambda x: x[0], prediction))

def formatForDf(index, result):
  return {'Id': index + len(X) + 1, 'SalePrice': result}

data = list(map(formatForDf, range(len(predicted_results)), predicted_results))
results_df = pd.DataFrame(data)

results_df.to_csv('lasso_linear_results.csv', index=False)