# Machine-Learning

1. Importing dataset
2. Modelling
3. Evaluation

## OOPS Concept

We have Classes, Objects and Methods in Machine Learning
- Libraries have different Modules
- Modules have different Classes
- We create objects for the classes
- Each object has different methods

### Example
```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:5] = sc.fit_transform(X_train[:, 3:5])
X_test[:, 3:5] = sc.transform(X_test[:, 3:5])
```


- preprocessing is the module of sklearn library
- standard Scaler is a class
- sc is an object of the class
- fit_transform is a method 

## Feature Scaling
- Applied to columns only
- To scale all the values in a particular range so that they become comparable
- Feature scaling should be applied after the splitting the dataset into training and test set because the test set is supposed to be a brand new set.( the test set should be fresh)
- Apply feature scaling on X_train and X_test separately
- *Feature Scaling should not be applied to dummy variables (dummuy variables refer to the columns that are encoded)*

### 1. Normalization
   > X' = (X - min(X))/(max(X) - min(X))
   - min(X) : min of that column
   - recommended when we have normal distribution in most features
   - [0 ; +1]
### 2. Standardization
   > X' = (X - mean(X))/ standard deviation(X)
   - mean(X) : mean of that column
   - standard deviation(X) : sd of X
   -  works almost all the time
   - [-3 ; +3]

## Features and Dependent variables

- Features are the independent variables (x)
- Dependent variables are mostly present in the last column (y)

## Encoding
- Encoding means transforming the categorical columns to real values so that we can use them to train the model. (categorical columns refer to the columns with string values)

### 1. One Hot Encoder
 - Creates new column and assigns binary values in 0s and 1s
### 2. Label Encoder
- Converts the column into real values (like 0, 1, 2...) depending on the number of categories.

## Underfitting and Overfitting
- Underfitting: A situation where a ML model performs better when being tested than when learning
- Overfitting: A situation where a ML model learns only on one dataset and cannot adapt to any other
- Equalfitting: A situation where a ML model works perfectly well and can solve any challenge

# Data Pre-Processing
1. Importing the Libraries
2. Importing the dataset
3. Taking care of missing values
4. Encoding the categorical data
5. Splitting dataset into training anf test set
6. Feature Scaling

# Regression

## 1. Simple Linear Regression
 > y = b0 + b1X1
- y: Dependent variable
- b0: y_intercept(constant)
- b1: Slope coefficient
- X1: Independent variable

  ### *How to determine the best regression line?*
  b0, b1 such that: SUM ((yi - y'i)^2 ) is minimized
  Ordinary Least Squares Method is used
  
## 2. Multiple Linear Regression
> y = b0 + b1X1+ b2X2 + .... + bnXn

### Assumptions of Linear Regression
1. Linearity (Linear relationship between Y and X)
2. Homoscedasticity (Equal variance) - No cone shape in graph
3. Multivariate Normality (Normality of error distribution)
4. Independence of observations (no autocorrealtion) - Ex. Stock Market
5. Lack of Multicollinearity (Predictors are not correlated with each other)
6. The outlier check (an "extra")

### Dummy Variables
- We extend out dataset for categorical columns depending on the number of categories in that column
- We will include n-1 dummy variable columns where n is the number of categories in that column

### Dummy variable Trap
- A statistical modeling issue that occurs when too many dummy variables are created for a categorical variable, resulting in multicollinearity.
- Always omit one dummy variable

### Statistical Significance
- A result has statistical significance when a result at least as "extreme" would be very infrequent if the null hypothesis were true.
- A statistically significant test result (P ≤ 0.05) means that the test hypothesis is false or should be rejected. A P value greater than 0.05 means that no effect was observed.
- *A p-value measures the probability of obtaining the observed results, assuming that the null hypothesis is true. The lower the p-value, the greater the statistical significance of the observed difference. A p-value of 0.05 or lower is generally considered statistically significant.*

### Building A Model
5 Methods:
1. All-in
   - Prior knowledge or
   - You have to or
   - Preparing for Backward elimination
2. Backward Elimination
   1. Select signifacnce level to stay in the model ( SL = 0.05)
   2. Fit the full model with all possible predictors
   3. Consider the predictor with the highest P-value. If P > SL, go to 4, else go to FIN
   4. Remove the predictor (the one with highest P value)
   5. Fit model without this variable the go to STEP 3
   
   FIN: Model is ready
      
3. Forward Selection
   1. Select a significance level to enter the model (SL = 0.05)
   2. Fit all regression models y ~ xn Select the one with the lowest P-value
   3. Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have
   4. Consider the predictor with the lowest P-value. if P < SL (good), go to STEP 3, otherwise go to FIN
   
   FIN: Keep the previous model
4. Birdirectional Elimination (Stepwise regression)
   1. Select the siginificance level to enter and to stay in the model (SLenter = 0.05, SLstay = 0.05)
   2. Perform the next step of Forward selection (new variables must have P < Senter to enter)
   3. Perform ALL steps of Backwad elimination (old variables must have P < SLstay to stay)
   4. No new variables can enter and no old variables can exit
   
   FIN: Your model is ready
5. Score Comparision
   1. Select criterion of goodness of fit (eg: Akaike criterion)
   2. Construt all possible regression models: 2^n - 1 total combinations
   3. Select the one with the best criterion
  
      *10 col means 1023 models*

*Method 2, 3 & 4 are Stepwise Regression*

#### *In Multiple Linear Regression we need not apply Feature Scaling because the coefficients do the work of balancing*

#### *The multiple linear regression class takes care of the dummy variable trap as well as the best features to choose for best significatnt value*


### 3. Polynomial Linear Regression
> y = b0 + b1x1 + b2x1^2 + ... + bnx1^n

 *It is still called 'Linear' because the coefficients are linear*



### 4. Support vector for regression (SVM)
### 5. Decision Tree Regression
### 6. Random Forest Regression



