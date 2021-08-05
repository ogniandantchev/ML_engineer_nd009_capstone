
## Machine Learning Engineer Nanodegree
## Udacity


# Capstone Project
# Report



## Ognian Dantchev
## August 5th, 2021



## Table of Contents

1. [Definition](#1-definition)

2. [Analysis](#2-analysis)

3. [Methodology](#3-methodology)

4. [Results](#4-results)

5. [Conclusion](#5-conclusion)

6. [References](#6-references)


## 1. Definition

### Project Overview

Time series forecasting is applied when scientific predictions are needed based on historical data with time component.  Time series forecasting has many applications in various industries.  Companies  use everything from simple spreadsheets to complex financial planning software in an attempt to accurately forecast future business outcomes such as product demand, resource needs, crops yield, traffic or financial performance.  Beyond business, applications include healthcare, medical, environmental studies, social studies and other forecasting.

There are many classical time series forecasting methods, based on autoregressive (AR) models, exponential smoothing, Fourier Transform, etc.

Models based on Neural Networks are able to handle more complex nonlinear patterns.  They have less restrictions and make less assumptions; have high predictive power and can be easily automated.  Their cons -- they require more data; can be more difficult to interpret, and, it is more difficult to derive confidence intervals for the forecast.   

In this project, three different Neural Networks were implemented.  A Recurrent Neural Network (RNN) with Gated Recurrent Unit (GRU) from the Keras API, as described in the TensorFlow Tutorials in [ [1](#6-references) ].  Second is a Dilated Causal CNN, I implemented earlier in [ [2](#6-references) ] and as originally described in [ [6](#6-references) ] by Aaron van den Oord, Sander Dieleman et al.  Both 1st and 2nd models use TensorFlow 2.5.  The third model implemented is based on the DeepAR+ algorithm of Amazon Forecast by AWS.


### Problem Statement

The purpose of this project is to implement model based on the DeepAR+ algorithm by Amazon and deploy it to AWS.  As a secondary target, I try to compare the results to other NN models for time series.  Own dataset with 45 years of local meteorology data was used, but no meteorology domain claims whatsoever.

The implementation was split into 5 smaller Jupyter Notebooks:

01. Generate and download the raw dataset at https://www.ncdc.noaa.gov/ .  Data import and exploration;  Changing some of the labels using the Inventory file.  Data cleaning and pre-processing;  Feature engineering and data transformation -- add day-of-year (between 1 and 366) and the hour-of-day (between 0 and 23);  Save the clean data in compact new format for use in next notebooks.

02. Split the data into training, validation, and testing sets;  Define and train a Gated Recurrent Unit (GRU) RNN Model.

03. Split the data into training, validation, and testing sets; Define and train a Dilated Causal CNN Model.

04. Connect AWS API session, create IAM role for Amazon Forecast, create S3 bucket; upload the dataset to AWS S3 bucket;  Define and train a model from AWS Amazon Forecast -- Amazon DeepAR+ algorithm.  I diviated from the original plan to use the open source Gluon Time Series (GluonTS) toolkit by AWS Labs, in favor of the Amazon DeepAR+ algorithm, as it is state-of-the-art and used by amazon.com; Deployed the later model to AWS

05. Discussion, TODO, Cleanup of AWS resources and References.  Evaluate and compare the models, to the extend possible.




### Metrics


...
> Making predictions about the future is called extrapolation in the classical statistical handling of time series data.  More modern fields focus on the topic and refer to it as time series forecasting.
> Forecasting involves taking models fit on historical data and using them to predict future observations. 

Time series forecasting derives from the much larger field of Time series analysis, https://en.wikipedia.org/wiki/Time_series 
Methods for time series analysis may be divided into two classes: frequency-domain methods and time-domain methods.  They may also be divided into linear and non-linear, and univariate and multivariate.



## 2. Analysis

pic
An attempt to predict the temperature for a given future period will be presented.  It will be based on data series for the temperature and the atmospheric pressure, for the target and three other cities in proximity.


## 3. Methodology

Weather data for the period 1931-2018 will be used for four cities in Bulgaria -- Ruse, Sofia, Varna and Veliko Tarnovo.  After struggling with the format of some of the local sources, I found that the National Centers for Environmental Information of the National Oceanic and Atmospheric Administration https://www.ncei.noaa.gov/ provide an access to a global source of climatic data.  It was also used in [2](#8-references), and one can pick almost any location in the world.  

Here's the sequence to acquire, clean and reshape the data:
* generate the raw data set at https://www.ncdc.noaa.gov/ (https://www7.ncdc.noaa.gov/CDO/cdoselect.cmd -- the old data request form)
* review the original dataset files
* resample data and helper functions
* remove data (where gaps are too big)
* add data (linearly interpolated from the neighboring values)
* save the clean, resampled data to a binary file.

This will be done in a separate Jupyter notebook, and then the clean data will be saved in NumPy .npy standard binary file format. 

The raw dataset is uploaded to GitHub for review: 
https://github.com/ogniandantchev/ML_engineer_nd009_capstone 


## 4. Results

### Model Evaluation and Validation

Recurrent neural networks (RNN) are a class of neural networks that is powerful for modeling sequence data such as time series or natural language.  The 1st model will be based RNN: Gated Recurrent Unit (GRU).  The 2nd one, I used in [ [2](#6-references) ] will be based on Dilated Causal CNN, often used for audio processing.

One of the models will then be deployed on AWS.

## 5. Conclusion

### Discussion 

An attempt was made to compare custom models -- Gated Recurrent Unit (GRU) RNN and the Dilated Causal CNN to Amazon's own Amazon DeepAR+ algorithm model.

+ The main goal was to learn how implement and deploy DeepAR+ model for time series forecast to AWS Amazon Forecast. 

+ The DeepAR+ model of Amazon Forecast (notebook 04) has RMSE of 2.806, while the custom Dilated Causal CNN (notebook 03) has an MSE 0.0037 (or RMSE = MSE **2 = 0.061).  Of course it is not fare to compare a model based on just one time series, to the multivariate custom model.

+ the total cost of training and deploying the model at AWS Amazon Forecast was EUR 0.78 (USD 0.91)



### Future improvements and TODO

+ implement a multivariate version of the predictor, based on the Amazon Forecast advanced examples [here](https://github.com/aws-samples/amazon-forecast-samples/tree/master/notebooks/advanced/Incorporating_Related_Time_Series_dataset_to_your_Predictor), and add the additional data from the originnal dataset -- athmospheric pressure and temperature for other cities in the region

+ at the moment metric for the model in 02 is MAE, in 03 metrics are MSE and MAPE, while Amazon Forecast uses RMSE and WAPE.  need to have same for all 3 models.

+ DeepAR+ of Amazon Forecast gives probabilistic Monte Carlo type evaluations, calculating P10, P50 and P90 -- the plot at the end of Notebook 04.  I.e., a statistical confidence level for an estimate. TODO: inderstand the concept and implement for the first two models.



----



###  Evaluation Metrics

The re-sampled data will be used to create "future" period by shifting the target-data. 

Mean Squared Error (MSE) will be used as the loss-function that will be minimized and used as a metric. This measures how closely the model's output matches the true output signals.

At the beginning of a sequence, the model has only seen input-signals for a few time-steps, so its generated output may be very inaccurate.  Therefore the model will be given a "warmup-period" of 50 time-steps where we don't use its accuracy in the loss-function, in hope of improving the accuracy for later time-steps, see [ [2](#8-references) ]




## 6. References


1. Magnus Erik Hvass Pedersen, [TensorFlow-Tutorials](http://www.hvass-labs.org/)
/ [GitHub repo](https://github.com/Hvass-Labs/TensorFlow-Tutorials) / [Videos on YouTube](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ)

2. Ognian Dantchev, Multivariate Time Series Forecasting with Keras and TensorFlow, [GitHub repo](https://github.com/ogniandantchev/dilated_causal_cnn_time_series)

3. Amazon Forecast resources, [AWS website](https://aws.amazon.com/forecast/resources/)

4. Time Series Forecasting Principles with Amazon Forecast, [Technical Guide](https://d1.awsstatic.com/whitepapers/time-series-forecasting-principles-amazon-forecast.pdf)

5. AWS Samples-- Amazon Forecast Samples [GitHub repo](https://github.com/aws-samples/amazon-forecast-samples)

6. Aaron van den Oord, Sander Dieleman et al., [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/pdf/1609.03499.pdf)




[1]: http://www.hvass-labs.org/

[2]: https://github.com/ogniandantchev/dilated_causal_cnn_time_series


[6]: https://arxiv.org/pdf/1609.03499.pdf


