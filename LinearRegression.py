'''
This program uses Linear Regression Trees algorithm to predict the medical prices.
'''

import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold


pd.options.mode.chained_assignment = None

# Load data
df_old= pd.read_csv("Totally_PreProcessedData.csv")
df = df_old[["DRG Definition", "Region_Numeric_Values1", "Total Discharges",
             "Median House Price 2011", "Average Total Payments"]]

# Features
feature_names= df[["DRG Definition", "Region_Numeric_Values1", "Total Discharges",
             "Median House Price 2011"]]

#Label
y=df["Average Total Payments"]

# Splitting data into train and test
kf = KFold(n_splits=7,shuffle=True)
kf.get_n_splits(feature_names)
print(kf)
for train_index, test_index in kf.split(feature_names):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = feature_names.iloc[train_index], feature_names.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# Create linear regression object
lr_model = linear_model.LinearRegression()

# Train the model using the training sets
model = lr_model.fit(x_train, y_train)

# Make predictions using the testing set
predictions = model.predict(x_test)


# Converting numpy array into dataframe
df_predictions=pd.DataFrame(predictions)


# Dataframe of y_test and df_predictions

df_output= pd.concat([y_test.reset_index(drop=True),df_predictions.reset_index(drop=True)],axis=1)
print(df_output)

#Converting DF to list

linearRegressionList= df_output.values.tolist()
print(linearRegressionList)

#Calculating mean absolute error of actual label value and predicted value
absDiffList=[]
for row in linearRegressionList:
    diff=abs(row[0]-round(row[1]))/row[0]
    absDiffList.append(diff)


sum=0
for i in range(0,len(absDiffList)):
    sum=sum+absDiffList[i]

    MAE=sum/len(absDiffList)

print("\nMean absolute error of a model is:\n")
print(MAE)