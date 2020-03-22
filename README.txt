


This program reads through a training dataset and allows the user to identify
the best course of action for predicting housing prices of the test dataset.

Lines 3-16: Required Packages to complete analysis

Lines 19-51: Clean train dataset. Resulted in some increased accuracy, apply same methods to test dataset
            note that the test dataset did not remove the houses > 4000 sf, as the full data should be read in the test
    
The program then runs through my through process and decision in predicitng the house prices.

Lines 60-65: Produce a list of data points that have a 70% correlation or greater, returned only two results
            Note that the results only have a 58% correlation with each other, this shows that there are more variables contribute
            in some meaningful way, out of the 79 variables total
    
    
Lines 75-142: The Variables class runs different models using the two most
            valuable variables, Overall Quality and Gross Living Area. This
            module runs regression model and then calculates KFolds, as well as
            scipy.stats and accuracy scores
            (quality = Variables(variables[0]) - line 230
            livingarea = Variables(variables[1])) - line 231

            
    Regression: This forms a regression model for both variables, predicting
                the Sales Price. The degree does not impact Gross Living Area,
                as the increase in r-squared was minimal. After moving to Degree 2,
                the r-square of Overall Quality did not change enough to warrant
                chasing r-squared with degree increases. As such, i stuck with
                Degree 2 for Gross Living Area and Degree 1 for Overall Quality
                Called on lines 234 and 235
                
    KMeans: This produces an elbow method chart for the model so the user knows
            which KMeans to select. This was more of a learning exercise than 
            anything, as the data from the KMeans showed a clear divide  at
            Overall Quality level 6 and split the Gross Living Area data in 
            quadrants near the center. This would have been more valuable if we
            really needed to dig in and find the cause behind the clusters, but
            it won't necessarily help us predict Housing Prices. Instatiated
            on lines 238 and 239
                            
                            
                            
Lines 145-202:  These lines run MLR, which produced better results than the previous
                models.
                
    MLR:        Runs MLR with 36 variables compared to the Y (Sale Price). Produces
                accuracy results, foremost RMSLE, which is what what Kaggle uses to
                prove accuracy. Called on line 244
            
    Test_MLR:   Runs MLR with the test dataset, returns to the calling variable
                to be exported to the submission CSV file. Called on line 249
                
Lines 205-225:  Run KMeans for Sale Price
                
NOTES:
If the user wants to change the correlation used to identify variables, change
0.70 to your desired correlation score on lines 70 and 71

If the user wants to change degrees used in the Linear Regression model, 
change the integer in the call variable (234 and 235) to the desired degree level

Can change K in KMeans by changing integer on lines 238-240

Submission to Kaggle has taken place via codes 251-256, code blocked to avoid 
unnecessary updating of submission file


            
    
