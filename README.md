# Time Series Forecasting

### compare GRU RNN, Dilated Causal CNN and Amazon Forecast DeepAR+ models

--- 


ToC / Jupyter notebooks:

01 Prepare the data

02 GRU RNN model

03 Dilated Causal CNN model

04 Amazon Forecast DeepAR+

05 Cleanup AWS, discussion, TODO and references.


-----

## References


1. Magnus Erik Hvass Pedersen, [TensorFlow-Tutorials](http://www.hvass-labs.org/)
/ [GitHub repo](https://github.com/Hvass-Labs/TensorFlow-Tutorials) / [Videos on YouTube](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ)

2. Ognian Dantchev, Multivariate Time Series Forecasting with Keras and TensorFlow, [GitHub repo](https://github.com/ogniandantchev/dilated_causal_cnn_time_series)

3. Amazon Forecast resources, [AWS website](https://aws.amazon.com/forecast/resources/)

4. Time Series Forecasting Principles with Amazon Forecast, [Technical Guide](https://d1.awsstatic.com/whitepapers/time-series-forecasting-principles-amazon-forecast.pdf)

5. AWS Samples-- Amazon Forecast Samples [GitHub repo](https://github.com/aws-samples/amazon-forecast-samples)




---

## Discussion

+ The main goal was to learn how implement and deploy DeepAR+ model for time series forecast to AWS Amazon Forecast. 

+ The DeepAR+ model of Amazon Forecast (notebook 04) has RMSE of 2.806, while the custom Dilated Causal CNN (notebook 03) has an MSE 0.0037 (or RMSE = MSE **2 = 0.061).  Of course it is not fare to compare a model based on just one time series, to the multivariate custom model.

+ the total cost of training and deploying the model at AWS Amazon Forecast was EUR 0.78 (USD 0.91)

---

## TODO:

+ implement a multivariate version of the predictor, based on the Amazon Forecast advanced examples [here](https://github.com/aws-samples/amazon-forecast-samples/tree/master/notebooks/advanced/Incorporating_Related_Time_Series_dataset_to_your_Predictor), and add the additional data from the originnal dataset -- athmospheric pressure and temperature for other cities in the region

+ at the moment metric for the model in 02 is MAE, in 03 metrics are MSE and MAPE, while Amazon Forecast uses RMSE and WAPE.  need to have same for all 3 models.

+ DeepAR+ of Amazon Forecast gives probabilistic Monte Carlo type evaluations, calculating P10, P50 and P90 -- the plot at the end of Notebook 04.  I.e., a statistical confidence level for an estimate. TODO: inderstand the concept and implement for the first two models.

---

## Tech notes

1. TensorFlow 2.5 was used, and it currently works with NumPy 1.19.5, but not the latest 1.20.x.  In the latest working version, I have these:

`tf.__version__`  -->  '2.5.0'

`np.__version__`  -->  '1.19.5'


2. Both notebooks 02 and 03 now work with TensorFlow 2


3. Latest versions of `s3fs` and `boto3` were installed locally.  for notebook 04, AWS Toolkit for VS Code was also used.   



4. Some minor issues with Amazon Forecast examples:
there are several versions of the AWS `util` on GitHub with different Amazon Forecast examples and tutorials.  I have included here the one that works for me.  but even that had issues with the latest versions of `s3fs` or `boto3`.  The `create_bucket` function was included in the notebook 04 in full, so it overrides the rest with the same name.


5. NB: Python Picle Format (Pandas read_pickle) night not work from Python 3.7 to 3.8.  Need to install and import picle5:

`conda install -c conda-forge pickle5`

Note to self: do not mix environments / work from 2 different computers


6. Models from 02 and 03 were saved (Keras H5 format) and added to the repo, but not needed later on, if one executes from scratch.

---

