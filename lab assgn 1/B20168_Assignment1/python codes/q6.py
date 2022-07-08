# Name : Shubham Shukla
# Roll no : B20168
# Branch : Computer Science and Engineering
# Contact no : 8317012277

# question 6

# importing panda to read csv file data and matplotlib to plot outputs
import pandas as pd
import matplotlib.pyplot as plt

# reading data using pandas and storing it in database variable
database = pd.read_csv("pima-indians-diabetes.csv")

# making a list whose boxplot is to be made
find_plot_lst = ['pregs', 'plas', 'pres', 'skin', 'test', 'pedi', 'Age', 'BMI']

# plotting the boxplot for specified attributes
for i in find_plot_lst:
    plt.boxplot(database[i])
    plt.show()