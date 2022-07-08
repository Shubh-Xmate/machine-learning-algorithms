# Name : Shubham Shukla
# Roll no : B20168
# Branch : Computer Science and Engineering
# Contact no : 8317012277

# question 3

# importing panda to read csv file data
import pandas as pd

# reading data using pandas and storing it in database variable
database = pd.read_csv("pima-indians-diabetes.csv")

'''Part A'''

# making a list of attributes whose correlation is to be calculated with Age attribute
with_corr = ['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi', 'Age']

# making a dictionary of correlation data wrt Age
# so that we can output the found data in series format
corr_data = dict()

# Appending the different correlation data with Age data in above dictionary
for attribute in with_corr:
    corr_data[attribute] = database['Age'].corr(database[attribute])

'''Part B'''

# making a list of attributes whose correlation is to be calculated with BMI attribute
with_corr2 = ['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi', 'Age']

# Again making a dictionary of correlation data wrt BMI
# so that we can output the found data in series format
corr_data2 = dict()

# Appending the different correlation data with BMI data in above second dictionary 
for attribute in with_corr2:
    corr_data2[attribute] = database['BMI'].corr(database[attribute])

# printing the output of correlation data with Age and BMI respectively
print("\nCorrelation of Age with different attributes is : ")
print(pd.Series(corr_data))
print("\nCorrelation of BMI with different attributes is : ")
print(pd.Series(corr_data2))