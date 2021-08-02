# Time Series Forecasting

compare GRU RNN to Dilated Causal CNN to Amazon Forecast DeepAR+


01 Prepare the data

02 GRU RNN model

03 Dilated Causal CNN model

04 Amazon Forecast DeepAR+

05 Cleanup AWS, discussion, TODO


-----


The implementation will be split into 5 or 6 smaller Jupyter Notebooks:

a. Data import and exploration;  Data cleaning and pre-processing;  Feature engineering and data transformation;  Save the clean data in compact new format for use in other notebooks and upload to AWS bucket,

b. Split the data into training, validation, and testing sets;  Define and train a Gated Recurrent Unit (GRU) RNN Model,

c. Split the data into training, validation, and testing sets; Define and train a Dilated Causal CNN Model,

d. Split the data into training, validation, and testing sets; Define and train a model from  Gluon Time Series (GluonTS) toolkit by Amazon Science,

e. Evaluate and compare the models,

f. Deploy one of the models to AWS