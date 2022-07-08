'''
IC272 
Lab Assignment 2
Shubham Shukla
B20168
'''

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""Question 1"""

# # importing important packages
# import pandas as pd
# import matplotlib.pyplot as plt

# # reading csv data 
# df = pd.read_csv('landslide_data3_miss.csv')

# # Creating the series data for missing values according to attributes
# missing_values = df.isnull().sum()

# # plotting the bar graph for the missing data according to attributes using series data obtained above
# plt.bar(missing_values.index, missing_values.values, color = 'g')
# plt.xticks(rotation = 30)
# plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""Question 2"""

# # importing the important packages
# import pandas as pd
# import numpy as np

# # reading csv file
# df = pd.read_csv('landslide_data3_miss.csv')

# """Part (a)"""

# # storing initial no of rows
# initial_rows = df.shape[0]

# # dropping the nan values of stationid attribute
# df = df.dropna(subset=['stationid'])

# # Storing the final no of rows
# final_rows = df.shape[0]

# # by substracting the initial no of rows with initial no of rows we get total no of rows dropped
# # hence printing it
# print("Total number of rows dropped is : ", initial_rows - final_rows)

# """Part (b)"""

# # creating list variable to store the deleted row data 
# drop_list = list()

# # iterating through the dataframe to store the specified row
# for i in df.index:
#     if(df.loc[i].isnull().sum() >= len(df.columns)/3):
#         # appending the row to be deleted
#         drop_list.append(df.loc[i])
#         # deleting the row
#         df = df.drop([i], axis = 0)

# # printing the data we get

# print("\n\nThe data having equal to or more than one third of attributes with missing values : ")
# # making the dropped list data to a dataframe to for better representation
# dropped_data = pd.DataFrame(drop_list)
# print(dropped_data)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''Question 3'''

# # Creating a series of the missing data values by attribute
# print("\n\nCount of the values which are missing corresponding to attributes : ")
# missing_series = df.isnull().sum()
# print(missing_series)

# # printing the sum of the total values of the obtained series
# print("\n\nSum of total missing values in the current data : ", sum(missing_series.values))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''Question 4'''

# """Part (a)"""
# # importing the important packages
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # reading the csv file
# df = pd.read_csv('landslide_data3_miss.csv')
# df2 = pd.read_csv('landslide_data3_original.csv')

# # dropping the row which don't have stationid data
# df = df.dropna(subset=['stationid'])

# # Creating a series of mean data from dataframe for to fill in other non present values wrt attributes
# mean_data = df.mean()

# # filling the mean data 
# for att in mean_data.index:
#     df[att] = df[att].fillna(mean_data[att])

# # creating a function which will return a dataframe of mean, median, mode and std from the passed dataframe
# def create_database(data):
#     mean_ = data.mean()
#     med_ = data.median()
#     mode_ = data.mode(); mode_ = mode_.iloc[0, 2:]
#     std_ = data.std()
#     statistical_data = pd.DataFrame({'Mean': mean_, 'Median': med_, 'Mode': mode_, 'Std': std_})
#     return statistical_data

# # Creating a function which will return the RMSE value for the passes attribute
# def RMSE_val(att):
#     Na = 0
#     sq_sum = 0;
#     for i in df.index:
#         if(df.loc[i, att] - df2.loc[i, att] != 0):
#             Na += 1;
#             sq_sum += pow(df.loc[i, att] - df2.loc[i, att], 2)
#     return pow(sq_sum/Na, 0.5)
    
# # Creating a list of attributes whose RMSE values is to found out
# RMSE_att = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']

# # printing the statistical dataframe for filled in dataframe with the original one
# print("Statistical data filling with mean values in missing data : \n")
# data1 = create_database(df);print(data1);
 
# print("\nStatistical data of the original dataframe : \n")
# data2 = create_database(df2);print(data2)

# # Creating a dictionary variable
# ind_dict = dict()

# # Inputting the value of RMSE in dictionary
# for att in RMSE_att:
#     ind_dict[att] = RMSE_val(att)

# # Plotting the graph for RMSE values
# plt.bar(ind_dict.keys(), ind_dict.values(), color = 'r')
# plt.xticks(rotation = 30)
# plt.show()

# """Part (b)"""

# # again reading the data
# df = pd.read_csv('landslide_data3_miss.csv')
# df2 = pd.read_csv('landslide_data3_original.csv')

# # dropping rows for non found values of stationid
# df = df.dropna(subset=['stationid'])

# # filling the dataframe with linear interpolation
# df = df.interpolate(method = 'linear')

# print("Statistical data after using linear interpolation : \n")
# data1 = create_database(df);print(data1);
 
# print("\nStatistical data of the original dataframe : \n")
# data2 = create_database(df2);print(data2)

# # Creating a list of attributes whose RMSE values is to found out
# RMSE_att = ['temperature', 'humidity', 'pressure', 'rain', 'lightavgw/o0', 'lightmax', 'moisture']

# # Creating a dictionary variable
# ind_dict = dict()

# # Creating a function which will return the RMSE value for the passes attribute
# def RMSE_val(att):
#     Na = 0
#     sq_sum = 0;
#     for i in df.index:
#         if(df.loc[i, att] - df2.loc[i, att] != 0):
#             Na += 1;
#             sq_sum += pow(df.loc[i, att] - df2.loc[i, att], 2)
#     return pow(sq_sum/Na, 0.5)

# # Inputting the value of RMSE in dictionary
# for att in RMSE_att:
#     ind_dict[att] = RMSE_val(att)

# # Plotting the graph for RMSE values
# plt.bar(ind_dict.keys(), ind_dict.values(), color = 'r')
# plt.xticks(rotation = 30)
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''Question 5'''

# # making a series for the 1st, 2nd, 3rd quartile ranges of temperature and rain attribute
# temp_qurt = df['temperature'].quantile(q = [0.25, 0.50,  0.75])
# rain_qurt = df['rain'].quantile(q = [0.25, 0.50, 0.75])

# # creating dictionary to store outliers of temperature and rain attribute
# temp_outliers = dict()
# rain_outliers = dict()

# # iterating through the attribute's series to store temperature and rain attribute's outliers
# for i in df['temperature'].index:
#     val = df['temperature'][i]
#     if(val < (temp_qurt[0.25] - 1.5*(temp_qurt[0.75] - temp_qurt[0.25])) or val > (temp_qurt[0.75] + 1.5*(temp_qurt[0.75] - temp_qurt[0.25]))):
#         temp_outliers[i] = val
# for i in df['rain'].index:
#     val = df['rain'][i]
#     if(val < (rain_qurt[0.25] - 1.5*(rain_qurt[0.75] - rain_qurt[0.25])) or val > (rain_qurt[0.75] + 1.5*(rain_qurt[0.75] - rain_qurt[0.25]))):
#         rain_outliers[i] = val

# # printing the collected dictionary data
# print("\nTemperature Outliers are : \n");print("Total outliers : ", len(temp_outliers.values()));print(temp_outliers)
# print("\nRain Outliers are : \n");print("Total outliers : ", len(rain_outliers.values())); print(rain_outliers)

# # plotting the data 
# plt.boxplot(df['temperature'])
# plt.title('Boxplot for temperature')
# plt.show()

# plt.boxplot(df['rain'])
# plt.title('Boxplot for rain')
# plt.show()

# # Changing outliers with the median of the data 
# for i in df['temperature'].index:
#     val = df.loc[i, 'temperature']
#     if(val < (temp_qurt[0.25] - 1.5*(temp_qurt[0.75] - temp_qurt[0.25])) or val > (temp_qurt[0.75] + 1.5*(temp_qurt[0.75] - temp_qurt[0.25]))):
#         df.loc[i, 'temperature'] = temp_qurt[0.50]
# for i in df['rain'].index:
#     val = df.loc[i, 'rain']
#     if(val < (rain_qurt[0.25] - 1.5*(rain_qurt[0.75] - rain_qurt[0.25])) or val > (rain_qurt[0.75] + 1.5*(rain_qurt[0.75] - rain_qurt[0.25]))):
#         df.loc[i, 'rain'] = rain_qurt[0.50]
        
# # analysing the data again first put quartile values in variables
# temp_qurt2 = df['temperature'].quantile(q = [0.25, 0.50,  0.75])
# rain_qurt2 = df['rain'].quantile(q = [0.25, 0.50, 0.75])

# # again creating dictionary to store outliers of temperature and rain attribute
# temp_outliers2 = dict()
# rain_outliers2 = dict()

# # iterating through the attribute's series to store temperature and rain attribute's outliers
# for i in df['temperature'].index:
#     val = df['temperature'][i]
#     if(val < (temp_qurt2[0.25] - 1.5*(temp_qurt2[0.75] - temp_qurt2[0.25])) or val > (temp_qurt2[0.75] + 1.5*(temp_qurt2[0.75] - temp_qurt2[0.25]))):
#         temp_outliers2[i] = val
# for i in df['rain'].index:
#     val = df['rain'][i]
#     if(val < (rain_qurt2[0.25] - 1.5*(rain_qurt2[0.75] - rain_qurt2[0.25])) or val > (rain_qurt2[0.75] + 1.5*(rain_qurt2[0.75] - rain_qurt2[0.25]))):
#         rain_outliers2[i] = val


# # Now let's print the collected dictionary data
# print("\nTemperature Outliers are after changing the ouliers with median are : \n");print("Total outliers : ", len(temp_outliers2.values()));print(temp_outliers2)
# print("\nRain Outliers are after changing the ouliers with median are : \n");print("Total outliers : ", len(rain_outliers2.values())); print(rain_outliers2)

# # Now again plotting the data
# plt.boxplot(df['temperature'])
# plt.title('Boxplot for temperature(after changing the outliers to median)')
# plt.show()

# plt.boxplot(df['rain'])
# plt.title('Boxplot for rain(after changing the outliers to median)')
# plt.show()