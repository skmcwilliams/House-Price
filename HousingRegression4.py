import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
import seaborn as sbn

# read test, train files
train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)
train.drop(train.columns[0], axis =1, inplace=True)
test.drop(test.columns[0], axis =1, inplace=True)

# Check shape of test data
train.shape
train.head()
test.shape
test.head()

# Interpolate data
train = train.select_dtypes(include=[np.number]).interpolate()
train.isnull().sum() != 0
test.select_dtypes(include=[np.number]).interpolate()
test = test.select_dtypes(include=[np.number]).interpolate()

# define variables
x_train = train.iloc[:, :-1].values
X_test = test.values
y = np.squeeze(np.array(train[['SalePrice']]))

# standardize scale
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
X_test = sc_x.transform(X_test)

# apply feature selection
fs = SelectKBest(score_func=f_regression, k=5)
x_train = fs.fit_transform(x_train, y)
X_test = fs.transform(X_test)

# Check New data
train.shape
train.head()
test.shape
test.head()
train.describe()
test.describe()

def dist_plot():
    """Distribution of Sale Prices, skewed right due to large amount of homes
        selling between $100k and $200k"""
    plt.figure(figsize=(10, 7))
    plt.tight_layout()
    sbn.distplot(train[['SalePrice']])
    plt.title('SalePrice Distribution')
    print('Skewed Distribution: ', scipy.stats.skew(train[['SalePrice']]))
    print('General Data Points:\n', train.SalePrice.describe())
    return plt.show()


dist_plot()

regressor = XGBRegressor(n_estimators=5000, learning_rate=0.05,
                         n_jobs=4,early_stopping_rounds=5)
regressor.fit(x_train, y)
y_pred = abs(regressor.predict(x_train))
plt.figure(figsize=(10, 7))
plt.scatter(y, y_pred, edgecolors=(0, 0, 0))
plt.title("Sale Prices")
plt.xlabel("Actual Sale Prices")
plt.ylabel('Predicted Sale Price')
plt.show()

# Visualize Actuals vs. Predicted
preds = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
print(preds.head(25))

# calculate scores

print('\nRESULTS OF TRAIN  MODEL:')
scores = scipy.stats.linregress(y, y_pred)
print(scores)
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
print('Root Mean Squared Log Error:',
      np.sqrt(metrics.mean_squared_log_error(y, y_pred)))

# Perform XGB on test data
regressor.fit(x_train, y)
predictions = abs(regressor.predict(X_test))

# Create submit CSV
submit = pd.read_csv('sample_submission.csv')
submit['SalePrice'] = predictions
print('\nTest Predictions: \n', submit.head(20))
submit.to_csv('submission.csv', index=False)
