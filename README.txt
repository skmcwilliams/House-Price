This program cleans the dataset by interpolation, scaling, and feature selection based on 10 features.
The program then uses XGBoost to predict house prices and prints the test predictions to a CSV file.

SalePrice data is not normalized and logistic regression was the best fit in a training, but displayed high
variance when applied to the test dataset. XGB allows for the most accurate prediction while incorporating early stopping
to avoid wasting time, which allows for a lower learning rate and many iterations, which decreases bias towards the training data


            
    
