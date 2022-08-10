'''
IC272 DSIII
lab 5

B20168 
Shubham Shukla

8317012277
'''

# importing all the important packages
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

"""PART A"""

# reading the csv data
df = pd.read_csv('SteelPlateFaults-train.csv')
df2 = pd.read_csv('SteelPlateFaults-test.csv')

# dropping the mentioned attributes due to correlation problem
df = df.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis = 1)
df2 = df2.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400'], axis = 1)
df = df.iloc[:, 1:]; df2 = df2.iloc[:, 1:]

# splitting the data into two part on the basis of class
# splitting for train data
df_train_c0 = df.loc[df['Class'] == 0];df_train_c1 = df[df['Class'] == 1]
real_train_c0 = df_train_c0['Class']; real_train_c1 = df_train_c1['Class']
df_train_c0 = df_train_c0.iloc[:, :23]; df_train_c1 = df_train_c1.iloc[:, :23]


# splitting for test data
df_test_c0 = df2[df2['Class'] == 0];df_test_c1 = df2[df2['Class'] == 1]
real_test_c0 = df_test_c0['Class']; real_test_c1 = df_test_c1['Class']
df_test_c0 = df_test_c0.iloc[:, :23]; df_test_c1 = df_test_c1.iloc[:, :23]

# storing the test data class column value
real_class = df2['Class']

# dropping the class columns from df and df2
df = df.iloc[:, :23]; df2 = df2.iloc[:, :23]

# calculating the prior probability for train data 
# this prior is for class 0
prior = real_train_c0.shape[0]/(real_train_c0.shape[0] + real_train_c1.shape[0])

# iterating through a loop for degree = 2, 4, 8, 16
q_comp = (2,4,8,16); high_acc = 0; res_q = 2
for comp in q_comp:
    # making GMM model for class = 0
    gmm0 = GaussianMixture(n_components=comp, covariance_type='full', reg_covar = 1e-5) 
    gmm0.fit(df_train_c0)
    
    # making GMM madel for class = 1
    gmm1 = GaussianMixture(n_components=comp, covariance_type='full', reg_covar = 1e-5)
    gmm1.fit(df_train_c1)
    
    # getting respective likelihood wrt classes
    gmm_ac_0 = gmm0.score_samples(df2)
    gmm_ac_1 = gmm1.score_samples(df2)
    
    # creating a list variable to store the prediction
    pred = list()
    
    # iterating through each sample and storing the prediction
    for i in range(df2.shape[0]):
        if(prior*gmm_ac_0[i] > (1 - prior)*gmm_ac_1[i]): pred.append(0);
        else : pred.append(1);
    
    # storing confusion matrix and accuracy score
    matrix  = confusion_matrix(real_class, pred)
    acc_sc = round(accuracy_score(real_class, pred), 3)
    
    print("The confusion matrix for Q = ", comp, " : \n", matrix)
    print("\nand the accuracy score is : ", acc_sc,"\n\n")
    
    if(high_acc < acc_sc): 
        high_acc = acc_sc
        res_q = comp
        
print(f"The highest accuracy is {high_acc} for {res_q} Q components.")

'''PART B'''

# reading the data 
df = pd.read_csv('abalone.csv')

# storing the target column data in a variable
target = df['Rings']

# dropping the target data from the dataframe
df = df.drop(columns = ['Rings'])

# Splitting the data in train test data
[X_train, X_test, tar_train, tar_test] = train_test_split(df, target, test_size=0.3, random_state=42, shuffle=True)

'''Question 1'''

# finding the attribute which has highest correlation coefficient with target attribute
max_corr = 0; name = None
for att in df.columns:
    curr_corr = abs(df[att].corr(target))
    if(curr_corr > max_corr):
        name = att
        max_corr = curr_corr
        
# making model on the basis of the highest correlated attribute
basis = X_train[name].values.reshape(-1, 1)
reg = LinearRegression().fit(basis, tar_train) 

'''part(a)'''

# let's make data input so that the prediction does not contain space
x = np.linspace(0, 1, len(X_train)).reshape(-1, 1)

# predicting the data for the made above prediction
pred = reg.predict(x)

# plotting the graph of the real and predicted one data
plt.scatter(basis, tar_train, color = 'b', s = 5, label = "real data")
plt.plot(x, pred, color = 'r', label = "best fit prediction")
plt.title("Showing best fit curve")
plt.xlabel("Shell weight")
plt.ylabel("Rings")
plt.legend()
plt.show()


'''part(b)'''
# predicting data to calculate rmse
pred_y_train = reg.predict(basis)

# rmse calculation
rmse_val = (mse(pred_y_train, tar_train))**0.5 
print("The rmse for the training data is : ", format(rmse_val, '.3f'))

'''part(c)'''
# making the test data input basis
basis2 = X_test[name].values.reshape(-1,1)

# predicting the output on the made basis for test data
pred_y_test = reg.predict(basis2)

# calculating rmse
rmse_val2 = (mse(pred_y_test, tar_test))**0.5
print("The rmse for the testing data is : ", format(rmse_val2, '.3f'))

'''part(d)'''
# plotting the graph of actual data and predicted one
plt.scatter(tar_test, pred_y_test, c = 'b', s = 5)
plt.xlabel('Actual rings')
plt.ylabel('Predicted rings')
plt.title('Actual vs predicted rings')
plt.show()

'''Question 2 '''

'''part(a)'''
# making multivariate regression data model
multi_reg = LinearRegression().fit(X_train, tar_train)

# predicting data on the basis of whole train data
pred = multi_reg.predict(X_train)

# calculating rmse
rmse_val = (mse(pred, tar_train))**0.5
print("The rmse for the training data is : ", rmse_val)

'''part(b)'''
# making prediction on test data on the basis of the model
pred2 = multi_reg.predict(X_test)

# calculating rmse value
rmse_val2 = (mse(pred2, tar_test))**0.5
print("The rmse for the testing data is : ", rmse_val2)

'''part(c)'''
# plotting the graph for prediction with actual rings data
plt.scatter(tar_test, pred2, c = 'b', s = 5)
plt.xlabel('Actual rings')
plt.ylabel('Predicted rings')
plt.title('Actual vs predicted rings')
plt.show()

'''Question 3'''

# making basis for the train and test data wrt most correlated attribute
basis1 = X_train[name].values.reshape(-1, 1)
basis2 = X_test[name].values.reshape(-1, 1)

# making a function which will make model according to the two data input, calculate the rmse and draw the plot internally
def prediction_accuracy(data1, data2):
    degree = (2,3,4,5) # polynomial degrees
    rmse_values = list() # to store the rmse data

    print("The rmse values for different degree are : ")
    for p in degree:
        # making object for changing linear data to some degree
        poly_trans = PolynomialFeatures(p) 
        
        # changing our data on the basis of polynomial degree
        x_ = poly_trans.fit_transform(data1)
        
        # making regression model
        poly_reg = LinearRegression()
        
        # fitting the changed dataframe to predict the values
        poly_reg.fit(x_, data2)
        
        # storing the prediction
        y_pred = poly_reg.predict(x_)
        
        # calculating rmse and appending it in list variable
        rmse_ = round((mse(y_pred, data2))**0.5, 3)
        rmse_values.append(rmse_)
        print("for degree =", p, ":", rmse_)

    # plotting the bar graph for rmse values wrt degree
    plt.bar(degree, rmse_values)
    plt.xlabel("Degree of polynomial")
    plt.ylabel("RMSE value")
    plt.title("Degree vs RMSE")
    plt.show()

'''part(a)'''
# calling the made function on the most correlated attribute and train data output
prediction_accuracy(basis1, tar_train)

'''part(b)'''
# calling the made function on the most correlated attribute and test data output
prediction_accuracy(basis2, tar_test)

'''part(c)'''
# since for higher degree we are getting less rmse values so we will choose p = 5 for best fit curve

# making object for changing polynomial features to degree = 5
poly_trans = PolynomialFeatures(5).fit_transform(basis1)

# making input data so that there will not be any gap in the predicted data(line)
for_line = np.linspace(0, 1, len(X_train)).reshape(-1, 1)

# transforming the make input data on the basis of degree 5
line_trans = PolynomialFeatures(5).fit_transform(for_line)

# making model 
poly_reg = LinearRegression()
poly_reg.fit(poly_trans, tar_train)

# predicting data
y_pred_train = poly_reg.predict(line_trans)

# plotting the graph
plt.scatter(basis1, tar_train, s = 5, color = 'b')
plt.plot(for_line, y_pred_train, linewidth=3, color='r')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Showing best fit curve')
plt.show()

'''part(d)'''
# making object of to transform our input data to the respective polynomial form for the prediction
poly_trans = PolynomialFeatures(5)

# transforming the data on the basis of above made object
x_test = poly_trans.fit_transform(basis2)

# making model
poly_reg = LinearRegression()
poly_reg.fit(x_test, tar_test)

# predicting data
y_pred_test = poly_reg.predict(x_test)

#plotting the graph for actual test output and predicted output
plt.scatter(tar_test, y_pred_test, s = 5, c = 'b')
plt.xlabel("Actual rings")
plt.ylabel("predicted rings")
plt.title("Actual vs predicted rings")
plt.show()

'''question 4'''

# making prediction on multivariate polymial regression data 
'''part(a)'''
# using the above made function for the multivariate train data
prediction_accuracy(X_train, tar_train)

'''part(b)'''
# using the above made function for the multivariate test data
prediction_accuracy(X_test, tar_test)

'''part(c)'''
# making object to change our data to the required polynomial form 
poly_trans = PolynomialFeatures(3)

# transforming our data to polynomial form 
x_test = poly_trans.fit_transform(X_test)

# making model to predict ring data
poly_reg = LinearRegression()
poly_reg.fit(x_test, tar_test)

# predicting data
y_test_pred = poly_reg.predict(x_test)

# plotting the graph between actual test data and predicted test data for attribute rings
plt.scatter(tar_test, y_test_pred, s = 5, c = 'b')
plt.xlabel("Actual rings")
plt.ylabel("predicted rings")
plt.title("Actual vs predicted rings")
plt.show()