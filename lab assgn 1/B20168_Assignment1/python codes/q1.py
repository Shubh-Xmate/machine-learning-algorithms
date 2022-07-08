# Name : Shubham Shukla
# Roll no : B20168
# Branch : Computer Science and Engineering
# Contact no : 8317012277

# question 1

# importing panda to read csv file data
import pandas as pd

# reading data using pandas and storing it in database variable
database = pd.read_csv("pima-indians-diabetes.csv")

# since we don't need class attribue hence we'll drop it
database = database.drop(['class'], axis = "columns")

# now using describe function we get a dataframe which gives details like
# ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
detail_data = database.describe()

# so we have to drop ['25%','50%','75%','count'] rows as we don't need it
detail_data = detail_data.drop(['25%','50%','75%','count'], axis = "rows")

# since mode and median data is not define by describe function 
# hence we have to make a dataframe for that and then append that dataframe to the detail_data dataframe

# Creating median data dictionary
med_data_lst =dict()
for name in database.columns:
    med_data_lst[name] = database[name].median()

# Creating mode data dictionary
mod_data_lst =dict()
for name in database.columns:
    # mode function gives values in the form of series(for mode of columns and rows) hence to take only value part, indexing is done
    mod_data_lst[name] = round(database[name].mode()[0], 3)

# Creating a second dataframe of median and mode to append this dataframe to first one to complete it
detail_data2 = pd.DataFrame([med_data_lst, mod_data_lst], index = ["median", "mode"])

# appending median, mode dataframe to detail_data
detail_data = detail_data.append(detail_data2)

# rounding every data to 3 decimal
detail_data.round(3)

# giving output in terminal
print(detail_data)