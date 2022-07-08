# '''
# IC272 DS3
# lab 4

# Shubham Shukla
# B20168
# '''
# # importing the important libraries
# import pandas as pd
# import sklearn as skit
# import numpy as np
# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler

# # reading the csv file
# df = pd.read_csv('SteelPlateFaults-2class.csv')

# # storing all data attribute except class
# x = df.iloc[:, :df.shape[1] - 1]

# # storing the class attribute data
# y = df.iloc[:, df.shape[1] - 1]

# """Question 1"""

# # Splitting the data in train test data
# [X_train, X_test, X_label_train, X_label_test] = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)

# # storing the train test data to external csv file
# X_train.to_csv('SteelPlateFaults-train.csv')
# X_test.to_csv('SteelPlateFaults-test.csv')

# print("Before normalizing the result are : \n")

# # making a loop that will compute confusion matrix, accuracy score for k = 1, 3, 5 respectively
# for i in range(3):
#     # using the inbuilt classifier
#     classifier = KNeighborsClassifier(n_neighbors = 2*i + 1, metric = 'minkowski', p = 2)
#     classifier.fit(X_train, X_label_train)
    
#     # storing the predicted data by classifier
#     y_pred = classifier.predict(X_test) 
    
#     # printing the output
#     print("\nFor K = ",2*i + 1 ," : ")
#     print("\nconfusion matrix is -> \n",confusion_matrix(y_pred, X_label_test))
#     print("\nAccuracy score is -> \n", accuracy_score(y_pred, X_label_test));
    
# """Question 2"""

# # making a funtion to normalize the data
# def normalize(data):
#     for att in data.columns:
#         data[att] = (data[att] - data[att].min())/(data[att].max() - data[att].min())
#     return data;

# # normalizing data and again splitting the data in train and test data
# x = normalize(x);
# [X_train, X_test, X_label_train, X_label_test] = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)

# print("\n\nAfter normalizing the result are : \n")

# # Again running a loop for k = 1, 3, 5 
# for i in range(3):
#     # using the inbuilt classifier
#     classifier = KNeighborsClassifier(n_neighbors = 2*i + 1, metric = 'minkowski', p = 2)
#     classifier.fit(X_train, X_label_train)
    
#     # storing the predicted data by classifier
#     y_pred = classifier.predict(X_test) 
    
#     # printing the output
#     print("\nFor K = ",2*i + 1 ," : ")
#     print("\nconfusion matrix is -> \n",confusion_matrix(y_pred, X_label_test))
#     print("\nAccuracy score is -> \n", accuracy_score(y_pred, X_label_test));
    
# '''Question 3'''

# # making a dataframe of the training data for my convenience
# X_train = (pd.DataFrame(X_train, columns = x.columns))
# X_train['Class'] = X_label_train
# df = X_train
# df['Class'] = X_label_train

# # splitting and storing the data wrt the classes
# data_class0 = df[df['Class'] == 0]; 
# data_class1 = df[df['Class'] == 1]

# # making a function that will find the covariance matrix 
# def create_cov(inp):
#     ind = inp.columns
#     lst = []
#     cov_list = []
#     for i in ind:
#         for j in ind:
#             cov_list.append(inp[i].cov(inp[j]))
#         lst.append(cov_list)
#         cov_list = []
#     return lst

# # storing mean vector and covariance vector wrt class
# data_class0 = data_class0.drop(['Class'], axis = 1); mean_class0 = data_class0.mean()
# data_class1 = data_class1.drop(['Class'], axis = 1); mean_class1 = data_class1.mean()
# cov_class0 = pd.DataFrame(create_cov(data_class0), index = data_class0.columns, columns = data_class0.columns)
# cov_class1 = pd.DataFrame(create_cov(data_class0), index = data_class0.columns, columns = data_class0.columns)

# # making a likelihood function viz going to return the expected unimodal values
# def likelihood(data_vector, mean_vector, cov_mat):
#     matrix = np.dot((data_vector-mean_vector).T, np.linalg.inv(cov_mat))
#     ins = -0.5*np.dot(matrix, data_vector - mean_vector)
#     ex = np.exp(ins)
#     return (ex/(abs(np.linalg.det(cov_mat)))**0.5)

# # making a function that will predict the class according to query vector data
# def class_predicted(data_vector):
#     p_0 = likelihood(data_vector, mean_class0, cov_class0)
#     p_1 = likelihood(data_vector, mean_class1, cov_class1)
#     if(p_0 > p_1): return 0;
#     else : return 1;
    
# # storing the predicted class in a list variable
# predicted_data = []
# for i in range(X_test.shape[0]):
#     predicted_data.append(class_predicted(X_test.iloc[i, :]))

# # printing the final output
# print("The statistics of predicted class by bayes classifier is : ")
# print("\nconfusion matrix is -> \n",confusion_matrix(predicted_data, X_label_test))
# print("\nAccuracy score is -> \n", accuracy_score(predicted_data, X_label_test))

# """Question 4"""
# # printing the mean data wrt classes
# print("The mean data are : ")
# print("\n for class 0 : \n", mean_class0)
# print("\n for class 1 : \n", mean_class1)

# # printing covariance data wrt classes
# print("The covariance matrix for class 0 : ")
# class0 = pd.DataFrame(cov_class0)
# print(class0)

# print("\nThe covariance matrix for class 1 : ")
# class1 = pd.DataFrame(cov_class1)
# class1