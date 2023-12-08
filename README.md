# decision-tree-model-for-regression-project
A standard implementation of a Decision Tree Regression using scikit-learn for a regression task

This code creates a Decision Tree Regressor model with a maximum depth of 5 using DecisionTreeRegressor() from scikit-learn. It then fits the model to the training data (X_train, y_train) using the fit() method.

After training the model, it uses the trained model to predict the target values for the test data (X_test) and calculates the Mean Squared Error (MSE) between the actual and predicted target values using metrics.mean_squared_error().

Finally, it prints the MSE, which evaluates the performance of the Decision Tree Regression model on the test data.
