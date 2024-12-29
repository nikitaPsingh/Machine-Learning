# Machine-Learning

1. Importing dataset
2. Modelling
3. Evaluation

## Feature Scaling
- Applied to columns only
- To scale all the values in a particular range so that they become comparable
- Feature scaling should be applied after the splitting the dataset into training and test set because the test set is supposed to be a brand new set.( the test set should be fresh)

1. Normalization
   X' = (X - min(X))/(max(X) - min(X))
   - min(X) : min of that column
2. Standardization
   X' = (X - mean(X))/ standard deviation(X)
   - mean(X) : mean of that column
   - standard deviation(X) : sd of X

## Features and Dependent variables

- Features are the independent variables (x)
- Dependent variables are mostly present in the last column (y)

## Encoding
- Encoding means transforming the categorical columns to real values so that we can use them to train the model. (categorical columns refer to the columns with string values)

  1. One Hot Encoding
     - Creates new column and assigns binary values in 0s and 1s
  3. Label Encoder
     - Converts the column into real values (like 0, 1, 2...) depending on the number of categories.
