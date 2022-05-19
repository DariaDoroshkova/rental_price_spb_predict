### Report
#### Rental housing price predictions in Saint-Petersburg
The project is based on data from Yandex.Realty classified https://realty.yandex.ru containing real estate listings for apartments in St. Petersburg and Leningrad Oblast from 2016 till the middle of August 2018.

**Project goal**
* Building machine learning model for predicting prices for rented housing, since accurate price prediction can help to find fraudsters automatically and help Yandex.Realty users to make better decisions when buying and selling real estate

**Steps:**
* Data preprocessing
* Building and training predicting models
* Choose the model with the best metrics
* Create web-service application
* Create Docker container

#### Data preprocessing
- - -

Below I will show some statistics about the dataset and list a few of the steps I took to prepare the data.

Initial dataset consists from 171186 observations and 17 columns (features). Target feature is 'last_price'. Here is information abot the data.
![alt text](https://raw.githubusercontent.com/DariaDoroshkova/rental_price_spb_predict/main/Images/data_info.PNG)

The data set included both apartments for rent and for sale. Therefore, first, the data was filtered according to the necessary criteria: offer type - rent, location - St. Petersburg.

Data preprocessing included filling missing values, cleaning outliers by IQR method, checking the most expensive and cheapest offers for realism.

Let's look at the target variable after data preparation. 
![alt text](https://raw.githubusercontent.com/DariaDoroshkova/rental_price_spb_predict/main/Images/density_plot.PNG)
Price has right-skewed distribution. There seems to be a lot of outliers since the price gets to 600,000 rubles for a month's rent.

I build a boxplot in order to look at the distribution of a variable from the other side.

![alt text](https://raw.githubusercontent.com/DariaDoroshkova/rental_price_spb_predict/main/Images/price_boxplot.PNG)

Evidently data outliers need to be cleaned up. For this purpose, I applied the IQR method.

Next I plotted heatmap to analyze the correlation between variables.
![alt text](https://raw.githubusercontent.com/DariaDoroshkova/rental_price_spb_predict/main/Images/heatmap.PNG)

As the map shows, the 'living_area' should be excluded from the data set, as it is highly correlated with the variable 'area'.

Then I visualized the statistical dependencies of numerical variables with the target variable.
![alt text](https://raw.githubusercontent.com/DariaDoroshkova/rental_price_spb_predict/main/Images/replot.PNG)
Thanks to this plot, we can not only infer the trend of the dependence of some variable and the target variable, but it is also a way of visualizing the data, which, for example, showed that some variables have outliers that are clearly errors, such as a kitchen area equal to 2500 sq m.

This is how the data is distributed over time. 
![alt text](https://raw.githubusercontent.com/DariaDoroshkova/rental_price_spb_predict/main/Images/time.PNG)

#### Building and training predicting models
- - -
After preparation the dataset consists of 126611 observations and 13 columns.
Here is sample of the data.
![alt text](https://raw.githubusercontent.com/DariaDoroshkova/rental_price_spb_predict/main/Images/data_sample.PNG)

In the first model, I used most of the variables and built it with RandomForestRegressor.

#### Features importance
To include only relevant variables in the model, feature_importance function was applied
![alt text](https://raw.githubusercontent.com/DariaDoroshkova/rental_price_spb_predict/main/Images/f_importance.PNG)

Features 'studio' and 'open plan' will not be used in further models, since there is no evidence of significance for the model.

I will build two models based on the following logic: the first model will include variables except for 'renovation' and 'agent fee'. Firstly, because these variables have a low contribution to the quality of the model, and secondly, these are important in terms of model implementation - data on renovation and agent fees are not always available.

#### Grid Search
To tune hyperparameters of RandomForestRegressor I applied GridSearch for both further models with following conditions
```python
param_grid = dict(
    n_estimators=[30, 50, 70, 100],
    max_depth=[5, 10, 20],
    )
grid_search_cv = GridSearchCV(estimator=model, 
                              param_grid=param_grid, 
                              cv=5,
                              verbose=5)
```
As a result I received the best parameters for regressors in each model

> grid_search_cv.best_params_
> 
> {'max_depth': 10, 'n_estimators': 100}

### Model 1
First model includes only four features
```python
features_1 = ['floor', 'area', 'kitchen_area', 'rooms']
```

```python
forest_1 = RandomForestRegressor(n_estimators=30, max_depth = 5, random_state=4)
```
As a result on train data the quality metrics of the model are as follows:
> RMSE on train 6046.11
> 
> MAPE on train 0.14%
> 
> MAE on train 3728.95
> 
> r2_score: 0.89

On test dataset quality metrics are slightly better:
> RMSE on test 5966.71
>  
> MAPE on test 0.14%
> 
> MAE on test 3644.20
> 
> r2_score: 0.9  

Holdout results are already the same as on train data: 

> RMSE on holdout 6655.54
> 
> MAPE on holdout 0.14%
> 
> MAE on holdout 4107.75
> 
> r2_score: 0.9  

### Model 2
In the second model more features were included
```python
features_2 = ['floor', 'area', 'kitchen_area', 'renovation', 'agent_fee', 'rooms']
```
```python
forest_2 = RandomForestRegressor(n_estimators=100, max_depth = 10, random_state=4)
```
On train data the quality metrics of the model are as follows:
> RMSE on train data 4809.04
> 
> MAPE on train data 0.10%
> 
> MAE on train data 2725.93
> 
> r2_score: 0.93 

On test dataset quality metrics are also slightly better:

> RMSE on test 4756.37
> 
> MAPE on test 0.09%
> 
> MAE on test 2668.40
> 
> r2_score: 0.94

Let's investigate quality metrics on holdout set:
> RMSE on holdout 5394.45
> 
> MAPE on holdout 0.10%
> 
> MAE on holdout 3115.55
> 
> r2_score: 0.93  

I consider the resulting quality metrics are good enough to use this model for price prediction.
A high R2 coefficient indicates that 93% of the change in the regressor is explained by variables from the model.

Variables 'renovation' and 'agent_fee' added to the second model are likely to be significant as the quality metrics improved after they were added.

### How to run Flask web application
- - -
Download source code models from repository.

Connect to your virtual machine. After this open special port in your remote machine using command:
```python
sudo ufw allow 5444
```
In Postman or another web service which you prefer crete GET request. 
Use IP address of your remote machine, then 5444 (port number) and name of the model 'predict_price'.
The toute will look as follows:
```python
51.250.110.142:5444/predict_price?floor=7&open_plan=0&rooms=1&st
```
Specify a list of parameters and their values. Click send and receive the result - price of apartment.

### How to run application using Docker
- - -

The Dockerfile is a text document containing all the commands to build the image.
In my project Dockerfile contains the following commands:
```python
from ubuntu:20.04 
MAINTAINER Daria Doroshkova
RUN apt-get update -y 
COPY . /opt/pythonproject
WORKDIR /opt/pythonproject
RUN apt install -y python3-pip 
RUN pip3 install -r requirements.txt 
CMD python3 ap.py
```

To run application with Docker, first, pull project
```python
docker pull dariadoroshkova/pythonproject:v.0.3
```
```python
docker run --network host -d dariadoroshkova/pythonproject:v.0.3
```
