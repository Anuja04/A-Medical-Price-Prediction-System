'''
This script combines two datasets 1. Medicare Payment 1. Housing Price using
and Creates one csv file with all necessary columns required from both datasets.
'''
# @author: Anuja Tike, SJSU, May 2018


import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

# Load original data
df= pd.read_csv("medicare_payment_2011.csv")
df1=pd.read_csv("Zip_MedianValuePerSqft_AllHomes.csv")

#Selecting DRG Definition,Provider State, Provider Zip Code,Hospital Referral Region Description, Total Discharges, Average Total Payments
df_medicarePayment=df[['DRG Definition','Provider State','Provider Zip Code','Hospital Referral Region Description',' Total Discharges ','Average Total Payments']]
df_medicarePayment=df_medicarePayment.rename(columns={' Total Discharges ':'Total Discharges' })

# Selecting only RegionID and 2011-12 columns from the dataframe
df_medianHousePrice=df1[['RegionID','2011-12']]

#Changing name of the column RegionID to Provider Zip Code and 2011-12 to Median House Price 2011
df_medianHousePrice=df_medianHousePrice.rename(columns={'RegionID':'Provider Zip Code','2011-12':'Median House Price 2011'})


#Merging medicare payments data and median house price data into one dataframe
df_master=pd.merge(df_medicarePayment,df_medianHousePrice,on='Provider Zip Code')


#Filling blank values in Median House Price 2011 column with NaN
df_master['Median House Price 2011'].replace('',np.nan,inplace=True)

#Dropping the rows having NaN values
df_master.dropna(inplace=True)

df_master['Median House Price 2011']=df_master['Median House Price 2011'].astype(int)

#Creating one master data csv file with all original features and label
df_master.to_csv("MasterData.csv", sep=',', encoding='utf-8')

