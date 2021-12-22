![Apache License](https://img.shields.io/hexpm/l/apa)  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  [![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)    ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)   ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![Made with matplotlib](https://user-images.githubusercontent.com/86251750/132984208-76ce70c7-816d-4f72-9c9f-90073a70310f.png)  ![seaborn](https://user-images.githubusercontent.com/86251750/132984253-32c04192-989f-4ebd-8c46-8ad1a194a492.png)  ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=Tableau&logoColor=white)  ![coursera](https://img.shields.io/badge/Coursera-0056D2?style=for-the-badge&logo=Coursera&logoColor=white) ![udemy](https://img.shields.io/badge/Udemy-EC5252?style=for-the-badge&logo=Udemy&logoColor=white)

## Flight Price Prediction

Airline companies use complex algorithms to calculate flight prices given various conditions present at that particular time. These methods take financial, marketing, and various social factors into account to predict flight prices.

Nowadays, the number of people using flights has increased significantly. It is difficult for airlines to maintain prices since prices change dynamically due to different conditions. That’s why we will try to use machine learning to solve this problem. This can help airlines by predicting what prices they can maintain. It can also help customers to predict future flight prices and plan their journey accordingly.


## Acknowledgements

 - [python for ML and Data science, udemy](https://www.udemy.com/course/python-for-machine-learning-data-science-masterclass)
 - [ML A-Z, udemy](https://www.udemy.com/course/machinelearning/)
 - [ML by Stanford University ](https://www.coursera.org/learn/machine-learning)



## Appendix

* [Aim](#aim)
* [Dataset used](#data)
* [Exploring the Data](#viz)
   - [Dashboard](#dashboard)
   - [Matplotlib](#matplotlib)
   - [Seaborn](#seaborn)
* [feature engineering](#fe)
* [prediction with various models](#models)
* [conclusion](#conclusion)

## AIM:<a name="aim"></a>

The objective is to predict flight prices given the various parameters. Data used in this article is publicly available at Kaggle. This will be a regression problem since the target or dependent variable is the price (continuous numeric value).

## Dataset Used:<a name="data"></a>

This dataset has been taken from [kaggle](https://www.kaggle.com/nikhilmittal/flight-fare-prediction-mh)

We have 2 datasets here — training set and test set.

The training set contains the features, along with the prices of the flights. It contains 10683 records, 10 input features and 1 output column — ‘Price’.

The test set contains 2671 records and 10 input features. The output ‘Price’ column needs to be predicted in this set. We will use Regression techniques here, since the predicted output will be a continuous value.

Following is the description of features available in the dataset –
1. Airline: The name of the airline.
2. Date_of_Journey: The date of the journey
3. Source: The source from which the service begins.
4. Destination: The destination where the service ends.
5. Route: The route taken by the flight to reach the destination.
6. Dep_Time: The time when the journey starts from the source.
7. Arrival_Time: Time of arrival at the destination.
8. Duration: Total duration of the flight.
9. Total_Stops: Total stops between the source and destination.
10. Additional_Info: Additional information about the flight
11. Price: The price of the ticket
## Exploring the Data:<a name="viz"></a>

I have used pandas, matplotlib and seaborn visualization skills.

**Matplotlib:**<a name="matplotlib"></a>
--------
Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, web application servers, and four graphical user interface toolkits.You can draw up all sorts of charts(such as Bar Graph, Pie Chart, Box Plot, Histogram. Subplots ,Scatter Plot and many more) and visualization using matplotlib.

* Environment Setup-->If you have Python and Anaconda installed on your computer, you can use any of the methods below to install matplotlib:

      pip: pip install matplotlib

      anaconda: conda install matplotlib
    
      import matplotlib.pyplot as plt

![matplotlib](https://eli.thegreenplace.net/images/2016/animline.gif)

for more information you can refer to [matplotlib](https://matplotlib.org/) official site

**Seaborn:**<a name="seaborn"></a>
------
Seaborn is built on top of Python’s core visualization library Matplotlib. Seaborn comes with some very important features that make it easy to use. Some of these features are:

**Visualizing univariate and bivariate data.**

**Fitting and visualizing linear regression models.**

**Plotting statistical time series data.**

**Seaborn works well with NumPy and Pandas data structures**

**Built-in themes for styling Matplotlib graphics**

**The knowledge of Matplotlib is recommended to tweak Seaborn’s default plots.**

Environment Setup==
If you have Python and Anaconda installed on your computer, you can use any of the methods below to install seaborn:

    pip: pip install seaborn

    anaconda: conda install seaborn
    
    import seaborn as sns
    
![seaborn](https://i.stack.imgur.com/uzyHd.gif)

for more information you can refer to [seaborn](https://seaborn.pydata.org/) official site.

**Dashboard:**<a name="dashboard"></a>
------

![flight_price](https://user-images.githubusercontent.com/86251750/145943856-1b61f1c5-76f0-471b-b3fd-07dd571ff8cf.png)

you can see the dashboard at [tableau](https://public.tableau.com/app/profile/pradeep7347/viz/flighpricepredictionDashboard/flight_price)

**Screenshots from notebook**

![download](https://user-images.githubusercontent.com/86251750/145944099-e2ca6b8b-ee92-4337-8953-503461ffbad5.png)

![download](https://user-images.githubusercontent.com/86251750/145944185-f7f3b722-0371-49e4-8056-3ea11a717ea4.png)

## My approaches on Feature Engineering<a name="fe"></a>
-------

* As there was very few missing values so I just drop them.
* Applyed pre-processing on duration column,Separate Duration hours and minute from duration
* Handling Categorical Data
      
      I used 2 main Encoding Techniques to convert Categorical data into some numerical format
      Nominal data --> data are not in any order --> OneHotEncoder is used in this case
      Ordinal data --> data are in order -->       LabelEncoder is used in this case
* extracted how many categories are in each cat_feature
* Outlier detection and handling the outliers
* seperate independent and dependent features
* Feature Selection

      Finding out the best feature which will contribute and have good relation with target variable. 
      Why to apply Feature Selection?
      To select important features to get rid of curse of dimensionality ie..to get rid of duplicate features
      
      I wanted to find mutual information scores or matrix to get to know about the relationship between all features.
      so I used Feature Selection using Information Gain.
* split dataset into train & test.
* dumping model using pickle so that we will re-use
* Finally used various regression model for predicion and done some hyperparameter tuning and choose the best model as my final model.

## Prediction with various Models:<a name="models"></a>
------

I have used various regression models for the prediction.

**RandomForestRegressor()**

    Training score : 0.9556692169211497

    r2 score: 0.7836751385244558
    MAE: 1220.6687220088443
    MSE: 4069813.514958583
    RMSE: 2017.3778810521799
    
![download](https://user-images.githubusercontent.com/86251750/145945709-8e99598b-b3e0-4a26-b922-5483f383b208.png)
    
**DecisionTreeRegressor()**
       
    Training score : 0.9685134197428378

    r2 score: 0.6741394214992276
    MAE: 1408.7951099672437
    MSE: 6130556.503440051
    RMSE: 2475.996062888641
![download](https://user-images.githubusercontent.com/86251750/145945907-5b1cb0fa-5fae-425f-9f60-1202bd38517e.png)

**LinearRegression()**

    Training score : 0.6218075684598243

    r2 score: 0.5899807190639135
    MAE: 1963.421195016378
    MSE: 7713870.701523291
    RMSE: 2777.3855874766996

**Hyperparameter Tuning**

`1.Choose following method for hyperparameter tuning`
`a.RandomizedSearchCV --> Fast way to Hypertune model`
`b.GridSearchCV--> Slow way to hypertune my model`

`2.Assign hyperparameters in form of dictionary`

`3.Fit the model`

`4.Check best paramters and best score`

* Random search of parameters, using 3 fold cross validation

**RandomizedSearchCV with RandomForestRegressor**

    r2 score : 0.8403085124679607
    MAE 1110.6771600647069
    MSE 3121555.2556893183
    RMSE 1766.792363490775
    
![download](https://user-images.githubusercontent.com/86251750/145946907-29482473-2b93-4a00-b7c3-82eb13136f16.png)

## CONCLUSION:<a name="conclusion"></a>
-----

From various model prediction we can see that RandomizedSearchCV with RandomForestRegressor give us the best performance.

So I choose RandomForest with RandomizedSearchCV as my final model.
