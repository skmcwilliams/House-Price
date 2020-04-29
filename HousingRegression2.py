import pandas as pd
import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, Lasso, ElasticNet
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
import seaborn as sbn


# read test, train files
train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

"""
# Check shape of test data
print(train.shape)
print(test.shape)
print(train.head())"""



# Drop Square footage over 4,000, as that is a large outlier
train.drop(train[train.GrLivArea > 4000].index, inplace = True)
train.drop(train[train.SalePrice > 450000].index, inplace = True)
train = train.select_dtypes(include=[np.number]).interpolate()
train.isnull().sum() != 0


x_train = train.iloc[:, :-1].values
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y = np.squeeze(np.array(train[['SalePrice']]))

fs = SelectKBest(score_func=f_regression, k=10)
# apply feature selection
x_train = fs.fit_transform(x_train, y)

# Train Dataframe shown below - copy to console"
x_train.shape
x_train


# Test Clean data by interpolating nulls, sum of nulls is now 0
test.select_dtypes(include=[np.number]).interpolate()
test = test.select_dtypes(include=[np.number]).interpolate()
X_test = sc_x.transform(test.values)


def dist_plot():
    """Distribution of Sale Prices, skewed right due to large amount of homes
        selling between $100k and $200k"""
    plt.figure(figsize = (10,7))
    plt.tight_layout()
    sbn.distplot(train[['SalePrice']])
    plt.title('SalePrice Distribution')
    print('Skewed Distribution: ', scipy.stats.skew(train[['SalePrice']]))
    print('Raw Data Points:/n', train.SalePrice.describe())
    return plt.show()

def elbow_method():
    """elbow method for KMeans"""
    model = KMeans()
    plt.figure(figsize=(10,7))
    visualizer = KElbowVisualizer(model, k=(1,15))
    visualizer.fit(train[['SalePrice']])
    visualizer.show()
    return plt.show()
    
            
"""
Create dictionary to save variables with relevant correlation to SalePrice,
then create list of those variables for easier iteration - Only two variables
have greater than a 70% correlation to SalePrice.
"""
correlations = {}
correlations.update(train[train.columns[0:]].corr('pearson')['SalePrice'])

variables = [variable for variable, correlation in correlations.items()
             if correlation >= 0.70 and correlation != 1.00
             or correlation <= -0.70 and correlation != -1.00]


# wanted to see correlation between Quality Rating and Sq Ft.
qual_sf_score = scipy.stats.linregress(np.squeeze(np.array(train[['OverallQual']])),
                            np.squeeze(np.array(train[['GrLivArea']])))
qual_sf_score



def predict(x, method, degrees = None, k = None):
      
    y = np.squeeze(np.array(train[['SalePrice']]))
    
    if method == 'regression':
        x = np.reshape(x,(-1, 1))
        poly = PolynomialFeatures(degree=degrees, include_bias=False)
        x = poly.fit_transform(x)
        regressor = LinearRegression()
    
    elif method == 'KMeans':
        """create KMeans clustering model"""
        # KMeans elbow method
        
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        predicted = kmeans.predict(x)
        plt.figure(figsize = (10,7))
        plt.scatter(x[:,0],x[:,1], c = predicted,
                    edgecolors=(0, 0, 0), cmap = 'rainbow')
        
        return plt.show()

    elif method == "MLR":
        # Build predictions and plot
        regressor = LinearRegression() 
               
    elif method == 'log':
        # Build predictions and plot
        regressor = LogisticRegression(penalty = 'l2', max_iter = 5000, n_jobs = 3)
                  
    elif method == 'SGD':
        # Build predictions and plot
        regressor = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, 
                 verbose=0, epsilon=0.1, random_state=None, 
                 learning_rate='invscaling', eta0=0.01, power_t=0.25,
                 early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, warm_start=False, average=False)
        
    elif method == 'Lasso':
        regressor = Lasso(alpha = 0.0001, max_iter = 5000)
        
    elif method == 'ElasticNet':
        regressor = ElasticNet(alpha = 0.0001)
        
    elif method == 'XGB':
        regressor = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

    regressor.fit(x, y) 
    y_pred = abs(regressor.predict(x))
    plt.figure(figsize = (10,7))
    plt.scatter(y, y_pred, edgecolors = (0, 0, 0))
    plt.title("Sale Prices")
    plt.xlabel("Actual Sale Prices")
    plt.ylabel('Predicted Sale Price')
    plt.show()
    
    # Visualize Actuals vs. Predicted
    preds = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
    print(preds.head(25))
    
    if method =='log':
        scores = accuracy_score(y, y_pred)
    else:
        scores = scipy.stats.linregress(y, y_pred)
    print('\nRESULTS OF TRAIN' + str(method) + ' MODEL:')
    print(scores)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))  
    print('Root Mean Squared Log Error:', np.sqrt(metrics.mean_squared_log_error(y, y_pred)))
    
    # Run KFolds
    scores = cross_val_score(regressor, x, y, cv=10)
    print('KFold Scores for Training Model:' + str(scores))
    mean = round(scores.mean()*100,2)
    print('Average Accuracy: ' + str(mean) + '%')

# dist_plot()
# elbow_method()



# predict(x_train, 'SGD')

# predict(x_train,'MLR')

    
# predict(x_train,'ElasticNet')

# predict(x_train,'Lasso')

# predict(x_train,'log')
predict(x_train, 'XGB')

    
# Perform on test data
"""
regressor = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

regressor.fit(x_train,y)
predictions = abs(regressor.predict(X_test))



#submit to kaggle
submit = pd.DataFrame()
#submit['id'] = test['id']
submit['SalePrice'] = predictions
print(submit.head(20))
submit.to_csv('sample_submission.csv', index = False)
"""