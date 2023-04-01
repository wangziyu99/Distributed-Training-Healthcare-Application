import pandas as pd
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle as pkl



x_train = pd.read_csv('data/diabetic_data.csv')
# local file
# x_train = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("hdfs:///home/ubuntu/cs230/spark/diabetic_data.csv")
x_train = x_train.drop(['encounter_id', 'patient_nbr'], axis=1)

x_train = x_train[[
    'num_lab_procedures', 'diag_1', 'diag_2', 'diag_3', 'num_medications', 'age', 
    'discharge_disposition_id', 'medical_specialty', 'time_in_hospital', 'num_procedures', 
    'readmitted']]

x_train.loc[x_train.age== '[0-10)','age'] = 0
x_train.loc[x_train.age== '[10-20)','age'] = 10
x_train.loc[x_train.age== '[20-30)','age'] = 20
x_train.loc[x_train.age== '[30-40)','age'] = 30
x_train.loc[x_train.age== '[40-50)','age'] = 40
x_train.loc[x_train.age== '[50-60)','age'] = 50
x_train.loc[x_train.age== '[60-70)','age'] = 60
x_train.loc[x_train.age== '[70-80)','age'] = 70
x_train.loc[x_train.age== '[80-90)','age'] = 80
x_train.loc[x_train.age== '[90-100)','age'] = 90
x_train.age = x_train.age.astype(np.int32)

categoricals = ['medical_specialty','diag_1', 'diag_2', 'diag_3']

for c in categoricals:
    x_train[c] = pd.Categorical(x_train[c]).codes

x_train.loc[x_train.readmitted != 'NO','readmitted'] = 0
x_train.loc[x_train.readmitted == 'NO','readmitted'] = 1

x_train.readmitted = x_train.readmitted.astype(np.int8)

y_train = x_train.readmitted
x_train = x_train.drop('readmitted', axis=1)

print(x_train.shape)


X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)


reg = xgb.XGBRegressor(tree_method="approx")
reg.load_model('regressor.json')

y_pred = (reg.predict(X_test) - 0.5 > 0).astype(int)

print(y_pred)
