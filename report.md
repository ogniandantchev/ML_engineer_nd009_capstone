
## Machine Learning Engineer Nanodegree
## Udacity


# Capstone Project
# Report



## Ognian Dantchev
## August 6th, 2021



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

In this project, three different Neural Networks were implemented for time series forecasting.  A Recurrent Neural Network (RNN) with Gated Recurrent Unit (GRU) from the Keras API, as described in the TensorFlow Tutorials in [ [1](#6-references) ].  Second is a Dilated Causal CNN, I implemented earlier in [ [2](#6-references) ] and as originally described in [ [6](#6-references) ] by Aaron van den Oord, Sander Dieleman et al.  Both 1st and 2nd models use Keras and TensorFlow 2.5.  The third model implemented is based on the DeepAR+ algorithm of Amazon Forecast by AWS.


### Problem Statement

The purpose of this project is to implement model based on the DeepAR+ algorithm by Amazon and deploy it to AWS.  As a secondary target, I try to compare the results to other NN models for time series.  Own dataset with 45 years of local meteorology data was used, but no meteorology domain claims whatsoever.

The implementation was split into 5 smaller Jupyter Notebooks:

01. Generate and download the raw dataset at https://www.ncdc.noaa.gov/ .  Data import and exploration;  Changing some of the labels using the Inventory file.  Data cleaning and pre-processing;  Feature engineering and data transformation -- add day-of-year (between 1 and 366) and the hour-of-day (between 0 and 23);  Save the clean data in compact new format for use in next notebooks.

02. Split the data into training, validation, and testing sets;  Define and train a Gated Recurrent Unit (GRU) RNN Model.

03. Split the data into training, validation, and testing sets; Define and train a Dilated Causal CNN Model. (model and model2 are the same, but use different way of building with Keras -- as an experiment)

04. Connect AWS API session, create IAM role for Amazon Forecast, create S3 bucket; upload the dataset to AWS S3 bucket;  Define and train a model from AWS Amazon Forecast -- Amazon DeepAR+ algorithm.  I diviated from the original plan to use the open source Gluon Time Series (GluonTS) toolkit by AWS Labs, in favor of the Amazon DeepAR+ algorithm, as it is state-of-the-art and used by amazon.com; Deployed the later model to AWS

05. Discussion, TODO, Cleanup of AWS resources and References.  Evaluate and compare the models, to the extend possible.




### Metrics



Common merics for Time Series Analysis are Mean Absolute Error (MAE), Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE),  Weighted Average Percentage Error (WAPE, also referred to as the MAD/Mean ratio.

$MSE=\sum_{j=0}^{n} (y_i - \hat{y_i})$

Mean Squared Error (MSE) will be used as the loss-function that will be minimized and used as a metric. This measures how closely the model's output matches the true output signals.

At the beginning of a sequence, the model has only seen input-signals for a few time-steps, so its generated output may be very inaccurate.  Therefore the model will be given a "warmup-period" of 50 time-steps where we don't use its accuracy in the loss-function, in hope of improving the accuracy for later time-steps, see [ [1](#6-references) ]

Amazon Forecast DeepAR+ uses RMSE and and WAPE metrics, [RMSD Wikipedia article](https://en.wikipedia.org/wiki/Root-mean-square_deviation)

I'm using this relation to comapre the metrics for the Dilated Causal CNN and the Amazon DeepAR+ model:
$RMSE=\sqrt{MSE}$


The re-sampled data will be used to create "future" period by shifting the target-data. 
---




## 2. Analysis

### Data Exploration

### Algorithms and APIs

![RNN](https://stanford.edu/~shervine/teaching/cs-230/illustrations/architecture-rnn-ltr.png?9ea4417fc145b9346a3e288801dbdfdc)


![Dilated Causal CNN, see 6](Dilated_Causal_CNN.png)



An attempt to predict the temperature for a given future period will be presented.  It will be based on data series for the temperature and the atmospheric pressure, for the target and three other cities in proximity.


## 3. Methodology


### Data Preprocessing

Weather data for the period 1931-2018 will be used for four cities in Bulgaria -- Ruse, Sofia, Varna and Veliko Tarnovo.  After struggling with the format of some of the local sources, I found that the National Centers for Environmental Information of the National Oceanic and Atmospheric Administration https://www.ncei.noaa.gov/ provide an access to a global source of climatic data.  It was also used in [ [1](#6-references)], and one can pick almost any location in the world.  

![Bulgaria Map](BG_c1.png)

Since weather data is most 

Here's the sequence to acquire, clean and reshape the data:
* generate the raw data set at https://www.ncdc.noaa.gov/ (https://www7.ncdc.noaa.gov/CDO/cdoselect.cmd -- the old data request form)
* review the original dataset files
* resample data and helper functions
* remove data (where gaps are too big)
* add data (linearly interpolated from the neighboring values)
* save the clean, resampled data to a binary file.

This was done in Jupyter notebook 01, and then the clean data was saved in Python standard binary file format. 

The raw dataset was uploaded in ZIP to GitHub, for review, at the project proposal stage: 
https://github.com/ogniandantchev/ML_engineer_nd009_capstone 


### Implementation

#### Gated Recurrent Unit (GRU) RNN Model Implementation

```
model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))
model.add(Dense(num_y_signals, activation="tanh")) 
```

Loss function calculates MSE, but after ignoring some "warmup" part of the sequences:
`warmup_steps = 50`


```
model.summary()

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
gru (GRU)                    (None, None, 512)         798720    
_________________________________________________________________
dense (Dense)                (None, None, 1)           513       
=================================================================
Total params: 799,233
Trainable params: 799,233
Non-trainable params: 0
_________________________
```

#### Dilated Causal CNN Model Implementation


```
# Keras Sequential definition -- var model2

p='causal' 
# this only works in TF 1.12+ -- December 2018+
# 2021: works with TensorFlow 2.5 too

model2 = Sequential([
    
    Conv1D(filters=32, input_shape=( None, num_x_signals ), 
           kernel_size=2, padding='causal', dilation_rate=1),
    Conv1D(filters=32, kernel_size=2, padding=p, dilation_rate=2),
    Conv1D(filters=32, kernel_size=2, padding=p, dilation_rate=4),
    Conv1D(filters=32, kernel_size=2, padding=p, dilation_rate=8),
    Conv1D(filters=32, kernel_size=2, padding=p, dilation_rate=16),
    Conv1D(filters=32, kernel_size=2, padding=p, dilation_rate=32),
    Conv1D(filters=32, kernel_size=2, padding=p, dilation_rate=64),
    Conv1D(filters=32, kernel_size=2, padding=p, dilation_rate=128),
    Dense(128, activation=tf.nn.relu), 
    Dropout(.2),
    Dense(1, activation="tanh")     
])

model2.compile(Adam(), loss='mean_absolute_error')
```

Using Standard MSE as loss here.

```

model2.summary()

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_16 (Conv1D)           (None, None, 32)          416       
_________________________________________________________________
conv1d_17 (Conv1D)           (None, None, 32)          2080      
_________________________________________________________________
conv1d_18 (Conv1D)           (None, None, 32)          2080      
_________________________________________________________________
conv1d_19 (Conv1D)           (None, None, 32)          2080      
_________________________________________________________________
conv1d_20 (Conv1D)           (None, None, 32)          2080      
_________________________________________________________________
conv1d_21 (Conv1D)           (None, None, 32)          2080      
_________________________________________________________________
conv1d_22 (Conv1D)           (None, None, 32)          2080      
_________________________________________________________________
conv1d_23 (Conv1D)           (None, None, 32)          2080      
_________________________________________________________________
dense_4 (Dense)              (None, None, 128)         4224      
_________________________________________________________________
dropout_2 (Dropout)          (None, None, 128)         0         
_________________________________________________________________
dense_5 (Dense)              (None, None, 1)           129       
=================================================================
Total params: 19,329
Trainable params: 19,329
Non-trainable params: 0
```

#### Amazon Forecast model, based on DeepAR+ algorithm

<img src="https://amazon-forecast-samples.s3-us-west-2.amazonaws.com/common/images/forecast_overview_steps.png" width="98%">





## 4. Results

### Model Evaluation and Validation

Recurrent neural networks (RNN) are a class of neural networks that is powerful for modeling sequence data such as time series or natural language.  The 1st model will be based RNN: Gated Recurrent Unit (GRU).  The 2nd one, I used in [ [2](#6-references) ] will be based on Dilated Causal CNN, often used for audio processing.

One of the models will then be deployed on AWS.

## 5. Conclusion

### Discussion 

An attempt was made to compare custom models -- Gated Recurrent Unit (GRU) RNN and the Dilated Causal CNN to Amazon's own Amazon DeepAR+ algorithm model.

+ The main goal was to learn how implement and deploy DeepAR+ model for time series forecast to AWS Amazon Forecast. 

+ The DeepAR+ model of Amazon Forecast (notebook 04) has RMSE of 2.806, while the custom Dilated Causal CNN (notebook 03) has an MSE 0.0037 (or $RMSE=\sqrt{MSE} = 0.061$).  Of course it is not fare to compare a model based on just one time series, to the multivariate custom model.

+ the total cost of training and deploying the model at AWS Amazon Forecast was EUR 0.78 (USD 0.91)



### Future improvements and TODO

+ implement a multivariate version of the predictor, based on the Amazon Forecast advanced examples [here](https://github.com/aws-samples/amazon-forecast-samples/tree/master/notebooks/advanced/Incorporating_Related_Time_Series_dataset_to_your_Predictor), and add the additional data from the originnal dataset -- athmospheric pressure and temperature for other cities in the region

+ at the moment metric for the model in 02 is MAE, in 03 metrics are MSE and MAPE, while Amazon Forecast uses RMSE and WAPE.  need to have same for all 3 models.

+ DeepAR+ of Amazon Forecast gives probabilistic Monte Carlo type evaluations, calculating P10, P50 and P90 -- the plot at the end of Notebook 04.  I.e., a statistical confidence level for an estimate. TODO: inderstand the concept and implement for the first two models.



----




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


