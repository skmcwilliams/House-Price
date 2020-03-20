"""
Steven McWilliams
IPD 237
Final Project
"""


import pandas as  pd
import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
from math import sqrt
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import time
import seaborn as sbn


# read test, train files
train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

# Check shape of test data
print(train.shape)
print(test.shape)



# Drop Square footage over 4,000, as that is a large outlier
train.drop(train[train.GrLivArea > 4000].index, inplace=True)

# Clean data by interpolating nulls, sum of nulls is now 0
cleantrain = train.select_dtypes(include=[np.number]).interpolate()


# CleanTrain Dataframe shown below - copy to console"
cleantrain.shape
cleantrain
print(sum(cleantrain.isnull().sum() != 0))


"""clean test data same as train data"""

# Clean data by interpolating nulls, sum of nulls is now 0
cleantest = test.select_dtypes(include=[np.number]).interpolate()
# Did not drop based on square footage for full data consumption

# CleanTest Dataframe shown below - copy to console
cleantest.shape
cleantest
print(sum(cleantest.isnull().sum() != 0))


"""
Create dictionary to save variables with relevant correlation to SalePrice,
then create list of those variables for easier iteration - Only two variables
have greater than a 70% correlation to SalePrice. Clustering done later on will
find relationships between these two variables - Ussed for Regression
"""
correlations = {}
correlations.update(cleantrain[cleantrain.columns[0:]].corr('pearson')['SalePrice'])

variables = [variable for variable, correlation in correlations.items()
             if correlation >= 0.70 and correlation != 1.00
             or correlation <= -0.70 and correlation != -1.00]


# wanted to see correlation between Quality Rating and Sq Ft.
qual_sf_score = scipy.stats.linregress(np.squeeze(np.array(cleantrain[['OverallQual']])),
                            np.squeeze(np.array(cleantrain[['GrLivArea']])))
print(qual_sf_score)

# Create Variable Class


class Variables:

    y = np.squeeze(np.array(cleantrain[['SalePrice']]))

    def __init__(self, x):
        self.x = np.squeeze(np.array(cleantrain[[x]]))

    def regressionplot(self, degrees):
        """Plot regression line"""
        x = np.reshape(self.x,(-1, 1))
        poly_features = PolynomialFeatures(degree=degrees, include_bias=False)
        X_poly = poly_features.fit_transform(x)
        model = LinearRegression()
        model.fit(X_poly,self.y)

        # Plot
        plt.figure(figsize=(10,7))
        X_plot_poly = poly_features.fit_transform(x)
        y_pred = abs(model.predict(X_plot_poly))
        plt.scatter(x, self.y, c ='b', )
        r2 = metrics.r2_score(self.y, y_pred)
        plt.plot(X_plot_poly, y_pred,'-r', label = 'R-Square: ' + str(r2))
        if len(np.unique([self.x])) > 11:
            plt.xlabel('Gross Living Area (sf)')
            plt.title('Gross Living Area vs. Sale Price ' + str(degrees) + ' Degree')
        else:
            plt.xlabel("Overall Quality")
            plt.title('Overall Quality vs. Sale Price ' + str(degrees) + ' Degree')
        plt.ylabel('Sale Price ($)')
        plt.legend()
        plt.show()
        scores = scipy.stats.linregress(self.y, y_pred)
        print(scores)
        print('RMSLE  of Linear Regression Model: %.2f'
              % sqrt(mean_squared_log_error(self.y, y_pred)))
    
        # Run KFolds for Regression
        scores = cross_val_score(model, x, self.y, cv = 10)
        print('KFold Scores for Linear Regression Model:' + str(scores))
        mean = round(scores.mean()*100,2)
        print('Average Accuracy: ' + str(mean) + '%')
        time.sleep(1)


    def KMeans(self, k):
        """create KMeans clustering model"""
        # KMeans elbow method
        x = np.reshape(self.x,(-1, 2))
        model = KMeans()
        plt.figure(figsize=(10,7))
        visualizer = KElbowVisualizer(model, k=(1,15))
        visualizer.fit(x)
        visualizer.show()
        plt.show()
        time.sleep(1)
        
        # KMeans graphs
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        predicted = kmeans.predict(x)
        plt.figure(figsize = (10,7))
        plt.scatter(x[:,0],x[:,1], c = predicted,
                    edgecolors=(0, 0, 0), cmap = 'rainbow')
        if len(np.unique([self.x])) > 11:
            plt.title('KMeans Gross Living Area')
        else:
            plt.title('KMeans Quality Rating')
        plt.show()


def MLR():
    """Because none of the models were accurate to be reach my goal of
    top 25% in the competition, leet's see if MLR is better"""
    # Create variables for MLR model
    MLR_X = cleantrain[[variable for variable in cleantrain if variable != 'SalePrice']].values

    MLR_Y = cleantrain['SalePrice'].values
    
    """Distribution of Sale Prices, skewed right due to large amount of homes
    selling between $100k and $200k"""
    plt.figure(figsize = (10,7))
    plt.tight_layout()
    sbn.distplot(MLR_Y)
    plt.show()
    print('Skewed Distribution: ', scipy.stats.skew(MLR_Y))
    
    # Build predictions and plot
    regressor = LinearRegression()
    regressor.fit(MLR_X, MLR_Y) 
    y_pred = abs(regressor.predict(MLR_X))
    plt.figure(figsize = (10,7))
    plt.scatter(MLR_Y, y_pred, edgecolors = (0, 0, 0))
    plt.title("Sale Prices")
    plt.xlabel("Actual Sale Prices")
    plt.ylabel('Predicted Sale Price')
    plt.show()
    time.sleep(1)
    # Visualize Actuals vs. Predicted
    preds = pd.DataFrame({'Actual': MLR_Y, 'Predicted': y_pred})
    print(preds.head(25))
 
    scores = scipy.stats.linregress(MLR_Y, y_pred)
    print(scores)
    print('\nRESULTS OF TRAIN MLR MODEL:')
    print('Mean Absolute Error:', metrics.mean_absolute_error(MLR_Y, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(MLR_Y, y_pred))  
    print('Root Mean Squared Log Error:', np.sqrt(metrics.mean_squared_log_error(MLR_Y, y_pred)))


    # Run KFolds for MLR
    scores = cross_val_score(regressor, MLR_X, MLR_Y, cv=10)
    print('KFold Scores for MLR Training Model:' + str(scores))
    mean = round(scores.mean()*100,2)
    print('Average Accuracy: ' + str(mean) + '%')
    

def Test_MLR():
    """Selected MLR, run on Test Data"""
    # Select Data for MLR 
    MLR_X = cleantrain[[variable for variable in cleantrain if variable != 'SalePrice']].values
    MLR_Y = cleantrain['SalePrice'].values
    X_Test = cleantest[[variable for variable in cleantest if variable != 'SalePrice']].values
    
    # Run predictions based on test data
    regressor = LinearRegression()
    regressor.fit(MLR_X, MLR_Y)
    y_test_pred = abs(regressor.predict(X_Test))
    return y_test_pred

# KMeans elbow method for Sale Price
model = KMeans()
plt.figure(figsize = (10,7))
visualizer = KElbowVisualizer(model, k = (1,15))
y = np.squeeze(np.array(cleantrain[['SalePrice']]))
y = np.reshape(y,(-1,2))
visualizer.fit(y)
visualizer.show()
plt.show()
time.sleep(1)

# KMeans graphs for Sale Price
kmeans = KMeans(n_clusters=4)
kmeans.fit(y)
predicted = kmeans.predict(y)
plt.figure(figsize = (10,7))
plt.scatter(y[:,0],y[:,1], c = predicted,
            edgecolors=(0, 0, 0), cmap = 'rainbow')
plt.title('KMeans Sale Price')
plt.show()

# Initiate Variables Class
quality = Variables(variables[0])
livingarea = Variables(variables[1])

# Call Regression, Degrees 2 for Quality performs best. No accurate enough for Kaggle
quality.regressionplot(2)
livingarea.regressionplot(1)

# Plot KMeans Charts
quality.KMeans(4)
livingarea.KMeans(4)




# Run MLR to see how this performs **best result in RMSLE and KFolds***
MLR()


"""Call Test_MLR as predictions to create list of predicted housing prices to be
pushed to dataframe and csv file"""
predictions = Test_MLR()

"""#submit to kaggle
submit = pd.DataFrame()
submit['SalePrice'] = predictions
print(submit.head(20))
submit.to_csv('sample_submission.csv', index = False)
"""