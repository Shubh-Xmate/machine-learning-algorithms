# Name : Shubham Shukla
# Roll no : B20168
# Branch : Computer Science and Engineering
# Contact no : 8317012277

# question 2

# importing important packages
import pandas as pd
import matplotlib.pyplot as plt

# reading data using pandas
database = pd.read_csv("pima-indians-diabetes.csv")

"""Part a"""

# plotting scatter graph wrt Age data for every specified attribute
for i in range(7):

    # creating a list of attributes which is required to be compared with BMI data
    name = ['pregs', 'plas', 'pres', 'skin', 'test', 'BMI', 'pedi']
    name2 = ['pregs', 'plas', ' pres (in mm Hg)', 'skin (in mm)', 'test (in mm U/mL)', 'BMI (in kg/m2)', 'pedi']

    # ploting graph wrt that attribute
    plt.scatter(database['Age'], database[name[i]], color = 'g', s = 7)
    plt.xlabel('Age (in years)')
    plt.ylabel(name2[i])
    plt.show()

"""Part b"""

# plotting scatter graph wrt Age data for every specified attribute
for i in range(7):

    # creating a list of attributes which is required to be compared with BMI data
    name = ['pregs', 'plas', 'pres', 'skin', 'test', 'pedi', 'Age']
    name2 = ['pregs', 'plas', ' pres (in mm Hg)', 'skin (in mm)', 'test (in mm U/mL)', 'pedi', 'Age (in years)']

    # ploting graph wrt that attribute
    plt.scatter(database['BMI'], database[name[i]], color = 'b', s = 5)
    plt.xlabel('BMI (in kg/m2) ')
    plt.ylabel(name2[i])
    plt.show()