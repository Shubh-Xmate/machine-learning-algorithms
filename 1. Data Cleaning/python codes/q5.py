# Name : Shubham Shukla
# Roll no : B20168
# Branch : Computer Science and Engineering
# Contact no : 8317012277

# question 5

# importing panda to read csv file data and matplotlib to plot outputs
import pandas as pd
import matplotlib.pyplot as plt

# reading data using pandas and storing it in database variable
database = pd.read_csv("pima-indians-diabetes.csv")

# refining data on the basis of class attribute and storing it in variables
data1 = database[database['class'] == 0]
data2 = database[database['class'] == 1]

# now below is code to draw histogram 

# for data1
# defining bin
bi = range(data1['pregs'].min(), data1['pregs'].max()+2, 1)

# code to plot histogram for given data
plt.hist(data1['pregs'], bins = bi, color = 'blue', edgecolor = 'y')
plt.grid(True)
plt.show()

# for data2
# defining bin
bi2 = range(data2['pregs'].min(), data2['pregs'].max()+2, 1)

# code to plot histogram for given data
plt.hist(data2['pregs'], bins = bi2, color = 'y', edgecolor = 'b')
plt.grid(True)
plt.show()