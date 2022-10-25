This script uses a logistic regression to classify types of irises using the [scikit-learn iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

**Logistic Regression**
- Predicts log-odds ratios of binary dependent variable from numeric or categorical independent variables.
- Classification is often performed if it is more likely that the obvervation is a member of the class than not (probability > 50%)
- Can calculate probabilities from log-odds ratios.

**Assumptions**
1. Binary response variable
2. Independent observations
3. Linearity of independent variables and log-odds
4. No multicollinearity among explanatory variables
5. No extreme outliers
6. Sample size is sufficiently large


```python
# Libraries for logistc regressions, plots, model testing, and robustness checks
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler #also consider RobustScaler to handle outliers
```


```python
# Configurable parameters
maximum_iterations = 25
random_state = 40
test_size = 0.2
high_vif = 5 
outlier_z = 3
```


```python
# Output parameters for readability
# pd.set_option('display.height', 10)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', 10)
```

**Loading, Manipulating and Exploring the Data**

This script uses the iris dataset so that anyone interested in replicating or modifying this script should be able to work with the example data easily.

Code for standardizing the data is included but commented out. Ultimately after creating a dataframe, I just look at a correlation matrix to help select variables later.

Because the iris dataset does not already split into test and training datasets, this section of the code also splits the data according to the test_size parameter in the configurable parameters section.


```python
#Data and data configurable parameters
# x_vars = ['sepal length (cm)','sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x_vars = ['petal length (cm)', 'sepal width (cm)']
y_var = 'target'

# Example data
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame( #concatenate data with target to make pandas dataframe
	data = np.c_[
	data['data'], data['target'] 
	],
	columns = data['feature_names'] + ['target']
	)

# The iris dataset has 3 classes and therefore is not directly suitable for a logistic regression so we will arbitrarily drop the 3rd class (only appropriate if this fits your data analysis goals)
df = df[df[y_var] != 0]
df[y_var] = pd.get_dummies(df[y_var])[1]

#Standardizing and/or normalization of features (optional)
# scaler = MinMaxScaler()
# df[x_vars] = scaler.fit_transform(df[x_vars])

# print(df)
# scaler = StandardScaler()
# df[x_vars] = scaler.fit_transform(df[x_vars])

#Print correlation matrix for predictors to help identify collinear variables
print(df[x_vars].corr(method = 'pearson',  min_periods = 1))

# For testing model performance, hold out a subset of the data for testing
training_data, testing_data = train_test_split(df,test_size = test_size, random_state = random_state)
```

                       petal length (cm)  sepal width (cm)
    petal length (cm)           1.000000          0.519802
    sepal width (cm)            0.519802          1.000000
    

**Fitting the Model**

Once the data is ready and the variables are selected, fitting the model is actually pretty simple.


```python
# Fit the model
logreg = LogisticRegression(random_state = random_state, max_iter = maximum_iterations)
logreg.fit(training_data[x_vars], training_data[y_var])
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(max_iter=25, random_state=40)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=25, random_state=40)</pre></div></div></div></div></div>



**Generating Predictions**

This section generates predictions for both the training and test datasets.


```python
# Return predictions
predicted_probabilities = logreg.predict_proba(testing_data[x_vars])
predicted_outcomes = logreg.predict(testing_data[x_vars])
```

**Evaluating Model Performance**

This section evaluates model performance using accuracy, true positive rate, false positive rate, true negative rate, false negative rate, precision, recall, and f1.


```python
# Evaluate model performance
cm = confusion_matrix = confusion_matrix(testing_data[y_var], predicted_outcomes)
true_negatives = cm[0][0]
false_negatives = cm[1][0]
false_positives = cm[0][1]
true_positives = cm[1][1]
tot = true_negatives + false_negatives + false_positives + true_positives

print('TN: ' + str(true_negatives) + ' FN: ' + str(false_negatives) + ' FP: ' + str(false_positives) + ' TP: ' + str(true_positives))
print('TNr: ' + str(true_negatives/(true_negatives+false_negatives)) + ' FNr: ' + str(false_negatives/(true_negatives+false_negatives)) + ' FPr: ' + str(false_positives/(true_positives+false_positives)) + ' TPr: ' + str(true_positives/(true_positives+false_positives)))

print(classification_report(testing_data[y_var], predicted_outcomes))
```

    TN: 8 FN: 2 FP: 0 TP: 10
    TNr: 0.8 FNr: 0.2 FPr: 0.0 TPr: 1.0
                  precision    recall  f1-score   support
    
               0       0.80      1.00      0.89         8
               1       1.00      0.83      0.91        12
    
        accuracy                           0.90        20
       macro avg       0.90      0.92      0.90        20
    weighted avg       0.92      0.90      0.90        20
    
    

**Model Assumption Checks**


```python
# Assumption checks: 1, binary response variable
if len(list(set(df[y_var]))) != 2:
	print("Warning: response variable is not binary")
```


```python
# Assumption 2, independence of observations: typically performed during data collection
```


```python
# Assumption 3, Linearity of independent variables and log-odds: box-tidwell test
df_logit = df.copy()
df_logit[x_vars] = np.log(df_logit[x_vars])
logit_results = GLM(df_logit[y_var], df_logit[x_vars], family=families.Binomial()).fit()

print(logit_results.summary()) #P-values less than .001 generally means non-linear relationship
```

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:                 target   No. Observations:                  100
    Model:                            GLM   Df Residuals:                       98
    Model Family:                Binomial   Df Model:                            1
    Link Function:                  Logit   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:                -56.310
    Date:                Tue, 25 Oct 2022   Deviance:                       112.62
    Time:                        11:42:10   Pearson chi2:                     105.
    No. Iterations:                     4   Pseudo R-squ. (CS):             0.2290
    Covariance Type:            nonrobust                                         
    =====================================================================================
                            coef    std err          z      P>|z|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    petal length (cm)    -7.5368      1.785     -4.223      0.000     -11.034      -4.039
    sepal width (cm)     11.1325      2.643      4.213      0.000       5.953      16.312
    =====================================================================================
    


```python
# Assumption 4, no multicollinearity: inspect variance inflation factor
if len(x_vars) != 1: #no need to test for multicollinearity with only 1 predictor variable
	vif = pd.DataFrame()
	vif['vif'] = [variance_inflation_factor(df[x_vars], i) for i in range(0,len(x_vars))] 
	vif['feature'] = x_vars
	high_vif_found = False
	for i in range(0,len(x_vars)):
		if vif['vif'][i] > high_vif:
			high_vif_found = True
			print("Warning: variance inflation factor for " + x_vars[i] + " is above " + str(high_vif))
	if high_vif_found:
		print(vif)
	else:
		print("Variance inflation factor for all independent variables is below " + str(high_vif))

```

    Warning: variance inflation factor for petal length (cm) is above 5
    Warning: variance inflation factor for sepal width (cm) is above 5
             vif            feature
    0  48.384252  petal length (cm)
    1  48.384252   sepal width (cm)
    


```python
# Assumption 5, no extreme outliers: find extreme x-values using internally studentized residuals
xvz = pd.DataFrame()
x_zscores = stats.zscore(df[x_vars])
x_scores_outliers = x_zscores > outlier_z
x_count_outliers = [sum(x_scores_outliers[x_vars[i]]) for i in range(0,len(x_vars))]
xvz['N high x zscores'] = x_count_outliers
xvz['feature'] = x_vars
high_xz_found = False
for i in range(0,len(x_vars)):
	if xvz['N high x zscores'][i] != 0:
		high_xz_found = True
		xi_outlier = [j for j in range(len(x_scores_outliers[x_vars[i]])) if x_scores_outliers[x_vars[i]][j]]
		print("Outliers found for " + x_vars[i] + ': ' + ", ".join(map(str, xi_outlier)))
		# print(xi_outlier)

if not high_xz_found:
	print("No outliers found")
else:
	print(xvz)
```

    No outliers found
    


```python
# Assumption 6 sufficient sample size is typically performed during data collection
```
