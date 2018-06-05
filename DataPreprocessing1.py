'''
This script transforms the columns in MasterData.csv in the required format by algorithms.
'''
# @author: Anuja Tike, SJSU, May 2018



import pandas as pd

pd.options.mode.chained_assignment = None

#Loading MasterData.csv
df=pd.read_csv('MasterData.csv')

#Converting Average Total Payments and DRG Definition to string for further processing
df['Average Total Payments']=df['Average Total Payments'].astype(str)
df['DRG Definition']=df['DRG Definition'].astype(str)

for i in range(0,len(df.axes[0])):
    # Removing '$' and ',' from Average Total Payments
    df['Average Total Payments'].iloc[i] = df['Average Total Payments'].iloc[i].replace('$',"").replace(',',"").replace(' ',"")

    #Extracting only DRG numbers from DRG Definition
    str=df['DRG Definition'].iloc[i]
    df['DRG Definition'].iloc[i]=str[0:3]

#Converting Average Total Payments and DRG definitions to float and integer respectively
df['Average Total Payments']=df['Average Total Payments'].astype(float)
df['Average Total Payments']=df['Average Total Payments'].astype(int)
df['DRG Definition']=df['DRG Definition'].astype(int)

#checking the unique values for column Hospital Referral Region Description
list1= list(df['Hospital Referral Region Description'].unique())

#Adding new column Region_Numeric_Values based on the column 'Hospital Referral Region Description'
df['Region_Numeric_Values1']=0

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='AK - Anchorage' ]= 1
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='AZ - Phoenix' ]=2
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='AZ - Mesa' ]= 3
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='AR - Springdale' ]=4

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='AR - Little Rock' ]= 5
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='MO - Springfield' ]= 9
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - Sacramento' ]= 14
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - Chico' ]= 15

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - Bakersfield' ]= 16
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - Los Angeles' ]= 17
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - San Diego' ]= 18

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - Orange County' ]= 19
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - Contra Costa County' ]= 20
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - San Luis Obispo' ]= 21

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - Redding' ]= 22
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - San Bernardino' ]= 23
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='NM - Albuquerque' ]= 30
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='ID - Boise' ]= 31

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Joliet' ]= 32
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Melrose Park' ]= 33
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Springfield' ]= 34
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Aurora' ]= 35

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Elgin' ]= 36
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Rockford' ]= 37
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Blue Island' ]= 38
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Peoria' ]= 39

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Chicago' ]= 40
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='MO - St. Louis' ]= 10
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='KY - Paducah' ]= 53
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Urbana' ]= 41

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IL - Hinsdale' ]= 42
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='KS - Wichita' ]= 43
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='MO - Kansas City' ]= 11

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='KS - Topeka' ]=44

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='LA - Lafayette' ]=45
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'LA - Houma']=46
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'LA - Baton Rouge']=47
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='LA - Alexandria' ]=48

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='LA - Metairie' ]=49
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'LA - Monroe']=50
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='MO - Columbia' ]=12
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'MO - Cape Girardeau' ]=13

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='MT - Billings' ]=54
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'NE - Omaha']=55
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'CO - Denver']=56
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='NV - Reno' ]=60
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'ND - Bismarck']=61

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='ND - Grand Forks' ]=62
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='OK - Oklahoma City' ]=63
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'OK - Tulsa']=64
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='OR - Medford' ]=65

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='OR - Portland' ]=66
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='OR - Eugene' ]=67
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='OR - Bend' ]=68
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'OR - Salem']=69
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='TX - San Antonio' ]=70

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='TX - Lubbock' ]=71
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='TX - Austin' ]=72
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'TX - Houston']=73
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'TX - Dallas']=74

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='TX - Corpus Christi' ]=75
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'UT - Provo']=88
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='UT - Ogden' ]=89
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='WA - Yakima' ]=91

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='WA - Seattle' ]=92
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='WA - Spokane' ]=93
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='WA - Tacoma' ]=94
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'WY - Casper']=96

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CO - Fort Collins' ]=57
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='AR - Fort Smith' ]=6
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'CA - San Francisco']=29

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='WI - Milwaukee' ]=97
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'LA - Shreveport']=51
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='TX - Waco' ]=76
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='HI - Honolulu' ]=98

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='AR - Jonesboro' ]=7
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'TX - Harlingen']=77
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'TX - Bryan']=78
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'TX - Beaumont']=79

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - San Jose' ]=24
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - Santa Barbara' ]=25
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'CA - Palm Springs/Rancho M']=26
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'CO - Colorado Springs']=58
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='MO - Joplin' ]=9
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='TX - Temple' ]=80
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='AR - Texarkana' ]=8

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='WA - Everett' ]=95
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'CO - Grand Junction']=59
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'TX - Victoria']=81
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'TX - McAllen']=82

df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='TX - Tyler' ]=83
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='IN - Evansville' ]=100
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='TX - Amarillo' ]=84
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'CA - Modesto']=27
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'TX - Odessa']=85
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='TX - Wichita Falls' ]=86
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'TN - Memphis']=99
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='TX - Abilene' ]=87
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']== 'UT - Salt Lake City']=90
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='CA - Fresno' ]=28
df['Region_Numeric_Values1'][df['Hospital Referral Region Description']=='LA - Lake Charles' ]=52


#Creating final csv file with all pre-processed data.
df_new=df[["DRG Definition","Region_Numeric_Values1","Total Discharges",
           "Median House Price 2011","Average Total Payments"]]

df_new.to_csv("Totally_PreProcessedData.csv",sep=',', encoding='utf-8')


