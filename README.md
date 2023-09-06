# Retail Sales Prediction 

### *Project Summary*
Rossmann is a chain drug store that operates in 7 European countries.In this project, we applied machine learning techniques to a real world problem of predicting stores sales.One of the biggest problems that every organisation has to deal with is predicting sales performance. To provide the correct product at the correct place and time, businesses must be able to foresee client demands.

Many parameters influencing sales are included in the data that we provide to our algorithm, such as store type, date, promotion, and so on. The objective is to predict 1115 stores’ daily sale numbers. Linear regression,Ridge regression,Lasso regression, Decision Tree, Random Forest, and XgBoost algorithms has been used to train models and predict sales.This kind of prediction enables store managers to take effective decisions and steps that increase productivity. We used feature selection, model selection to improve our prediction results. In view of the nature of our problem,we have used Root Mean Square Error(RMSE), Mean Absolute Error(MAE) and R2 Score to measure the accuracy of predictions.

Steps that we have followed include Dataset loading, Data Cleaning, Data Wrangling, Handling of Outliers, and Missing Values ,Exploratory data analysis, and Feature Engineering etc.Through EDA, we got information about different features that are affecting sales and sales patterns throughout the year. Insights found from exploratory data analysis will be helpful in feature selection. For the treatment of outliers, the IQR (Interquartile Range) method was adopted because our data is positively skewed distributed.In the Feature Engineering part, for the encoding of categorical variables,we have used Label Encoding and One Hot Encoding.To check the normality of the data, a Q-Q plot was used, and the best transformation for the feature was determined by using the Q-Q plot. Log Transformation and Square Root Transformation were used for feature transformation. To detect multicollinearity,we have used a correlation matrix and performed VIF analysis, and with these techniques, multicollinearity from the data is effectively removed. For the evaluation of model metrics like MSE, RMSE, and MAPE, were calculated. For further improvement in the model and to find the best set of hyperparameters, we have performed Hyperparameter tuning for each model, and the techniques utilised for this purpose are GridSearchCV and RandomizedSearchCV. To understand models better, a Model Explainability tool like SHAP has been used. To understand the model better, a model explainability tool like SHAP has been used. Local as well as Global explainability have been explained using SHAP values.

After the successful implementation of all the algorithms on training and test data, the obtained results show that Tree Based algorithms and Ensemble methods perform better than linear regression. At the end, it is found that the XGBoost algorithm predicts sales with high accuracy on training data and also gives good results on test data. That's why XGBoost Regression model has been chosen as the final model for our analysis.


### *Problem Statement*
Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied. You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.


### *Dataset Information*

* There are two datasets: the first one contains information about the sales of different Rossmann stores, and the second one contains information about stores.
* Sales data has 1017209 rows and 9 columns, whereas store data has 1115 rows and 10 columns.

#### *Variable Description*

* **Id** - an Id that represents a (Store, Date) duple within the test set
* **Store** - a unique Id for each store
* **Sales** - the turnover for any given day (this is what you are predicting)
* **Customers** - the number of customers on a given day
* **Open** - an indicator for whether the store was open: 0 = closed, 1 = open
* **StateHoliday** - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
* **SchoolHoliday** - indicates if the (Store, Date) was affected by the closure of public schools
* **StoreType** - differentiates between 4 different store models: a, b, c, d
* **Assortment** - describes an assortment level: a = basic, b = extra, c = extended
* **CompetitionDistance** - distance in meters to the nearest competitor store
* **CompetitionOpenSince[Month/Year]** - gives the approximate year and month of the time the nearest competitor was opened
* **Promo** - indicates whether a store is running a promo on that day
* **Promo2** - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating

* **Promo2Since[Year/Week]** - describes the year and calendar week when the store started participating in Promo2
* **PromoInterval** - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

### *Project Work flow*

1. Importing Libraries

2. Loading the Dataset

3. Explore Dataset

4. Data Wrangling

5. Exploratory Data Analysis (EDA) 

6. Hypothesis Testing

7. Feature Engineering & Data Pre-processing

8. Machine Learning Model Implementation

9. Model Explainability

10. Future Work

11. Conclusions



### *Python Libraries Used*

* Numpy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* SciPy
* Shap
* Pylab
