
## Machine Learning Engineer Nanodegree
## Udacity


# Capstone Project
# Proposal
# Multivariate Time Series Forecasting


## Ognian Dantchev
## July 2nd, 2021



## Table of Contents

1. [Domain Background](#1-domain-background)

2. [Problem Statement](#2-problem-statement)

3. [Datasets and Inputs](#3-datasets-and-inputs)

4. [Solution Statement](#4-solution-statement)

5. [Benchmark Models](#5-benchmark-model)

6. [Evaluation Metrics](#6-evaluation-metrics)

7. [Project Design](#7-project-design)

8. [References](#8-references)


## 1. Domain Background

Time series forecasting has many applications in many industries like: 
* Medical (EEG reading), 
* Healthcare (birthrate, hospital cases), 
* Retail (unit sales per peroduct per shop), 
* Agriculture (crops yield), 
* Finance (stock price), 
* Transportation (number of passengers), 
* IT (server utilization)

Time series forecasting is applied when scientific predictions are needed based on historical data with time component. [1](#8-references)


> Making predictions about the future is called extrapolation in the classical statistical handling of time series data.  More modern fields focus on the topic and refer to it as time series forecasting.

> Forecasting involves taking models fit on historical data and using them to predict future observations. [2](#8-references)






## 2. Problem Statement

An attempt to predict the temperature for a given future period will be presented.  It will be based on data series for the temperature and the atmospheric pressure, for the target and three other cities in proximity.

RNN

## 3. Datasets and Inputs

Weather data for the period 1931-2018 will be used for four cities in Bulgaria -- Ruse, Sofia, Varna and Veliko Tarnovo.  After struggling with the format of some of the local sources, I found that the National Centers for Environmental Information of the National Oceanic and Atmospheric Administration https://www.ncei.noaa.gov/ provide an access to a global source of climatic data.  It was also used in [1](#8-references), and one can pick almost any location in the world.  

Here's the sequience to aquire, clean and reshape the data:
* generate the raw data set at https://www.ncdc.noaa.gov/ (https://www7.ncdc.noaa.gov/CDO/cdoselect.cmd -- the old data request form)
* review the original dataset files
* resample data and helper functions
* remove data (where gaps are too big)
* add data (linearly interpolated from the neighbouring values)
* save the clean, resampled data to a binary file.

This will be done in a separate Jupyter notebook, and then the clean data will be saved in NumPy .npy standard binary file format. 

## 4. Solution Statement

https://www.tensorflow.org/guide/keras/rnn

## 5. Benchmark Model

Gluon Time Series (GluonTS) toolkit by Amazon Science

## 6. Evaluation Metrics



## 7. Project Design

5 or 6 smaller Jupyter Notebooks

a. Data import and exploration;  Data cleaning and pre-processing;  Feature engineering and data transformation;  Save the clean data in compact new format for use in other notebooks and upload to AWS bucket. 

b. Split the data into training, validation, and testing sets;  Define and train a Gated Recurrent Unit (GRU) RNN Model.

c. Split the data into training, validation, and testing sets; Define and train a Dilated Causal CNN Model.

d. Split the data into training, validation, and testing sets; Define and train a model from  Gluon Time Series (GluonTS) toolkit by Amazon Science

e. Evaluate and compare the models.

f. Deploy one of the models to AWS




## 8. References

1. 
https://www.tableau.com/learn/articles/time-series-forecasting

2. 
https://machinelearningmastery.com/time-series-forecasting/

3. Magnus Erik Hvass Pedersen, [TensorFlow-Tutorials][1]
/ [GitHub](https://github.com/Hvass-Labs/TensorFlow-Tutorials) / [Videos on YouTube](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ)



4. Aaron van den Oord, Sander Dieleman et al., [WaveNet: A Generative Model for Raw Audio][3]


5. Ognian Dantchev,  [Multivariate Time Series Forecasting with Keras and TensorFlow][4], GitHub repo




[3]: http://www.hvass-labs.org/


[4]: https://arxiv.org/pdf/1609.03499.pdf

[5]: https://github.com/ogniandantchev/dilated_causal_cnn_time_series




The project's domain background — the field of research where the project is derived;
A problem statement — a problem being investigated for which a solution will be defined;
The datasets and inputs — data or inputs being used for the problem;
A solution statement — the solution proposed for the problem given;
A benchmark model — some simple or historical model or result to compare the defined solution to;
A set of evaluation metrics — functional representations for how the solution can be measured;
An outline of the project design — how the solution will be developed and results obtained.



