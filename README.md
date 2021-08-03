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


## Tech notes

TensorFlow 2.5 was used, and it currently works with NumPy 1.19.5, but not the latest 1.20.x

`tf.__version__`  -->  '2.5.0'

`np.__version__`  -->  '1.19.5'


Both notebooks 02 and 03 now work with TensorFlow 2


Latest versions of `s3fs` and `boto3` were installed locally.  for notebook 04, AWS Toolkit for VS Code was also used.   



Some minor issues with Amazon Forecast examples:

there are several versions of the AWS `util` on GitHub with different Amazon Forecast examples and tutorials.  I have included here the one that works for me.  but even that had issues with the latest versions of `s3fs` or `boto3`.  The `create_bucket` function was included in the notebook 04 in full, so it overrides the rest with the same name.


NB: Python Picle Format (Pandas read_pickle) night not work from Python 3.7 to 3.8.  Need to install and import picle5:

`conda install -c conda-forge pickle5`


Models from 02 and 03 were saved (Keras H5 format) and added to the repo, but not needed later on, if one executes from scratch.

---

The implementation will be split into 5 or 6 smaller Jupyter Notebooks:

a. Data import and exploration;  Data cleaning and pre-processing;  Feature engineering and data transformation;  Save the clean data in compact new format for use in other notebooks and upload to AWS bucket,

b. Split the data into training, validation, and testing sets;  Define and train a Gated Recurrent Unit (GRU) RNN Model,

c. Split the data into training, validation, and testing sets; Define and train a Dilated Causal CNN Model,

d. Split the data into training, validation, and testing sets; Define and train a model from  Gluon Time Series (GluonTS) toolkit by Amazon Science,

e. Evaluate and compare the models,

f. Deploy one of the models to AWS