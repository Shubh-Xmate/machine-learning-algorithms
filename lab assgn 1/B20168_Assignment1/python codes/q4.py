# Name : Shubham Shukla
# Roll no : B20168
# Branch : Computer Science and Engineering
# Contact no : 8317012277

# question 4

# importing panda to read csv file data and matplotlib to plot outputs
import pandas as pd
import matplotlib.pyplot as plt

# reading data using pandas and storing it in database variable
database = pd.read_csv("pima-indians-diabetes.csv")

"""Part A"""
# We will draw histogram for pregnancy count data here

# defining bin for the histogram
bi = range(database['pregs'].min(), database['pregs'].max()+1, 1)

# defining other attributes for histogram and plotting it
plt.hist(database['pregs'], bins = bi, edgecolor = 'blue', color = 'yellow')
plt.grid(True)
plt.show()

"""Part B"""
# We will draw histogram for Triceps skin fold thickness data here

# defining bin for the histogram
bi2 = range(database['skin'].min(), database['skin'].max()+1, 1)

# defining other attributes for histogram and plotting it
plt.hist(database['skin'], bins = bi2, color = 'orange')
plt.grid(True)
plt.show()