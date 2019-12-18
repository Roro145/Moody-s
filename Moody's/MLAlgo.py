#Imports the necessary libraries for organizing data, training the neural network and creating sample students
import pandas as pds
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np


#using the pandas library to read the dataset for this problem
#The dataset consists of 12 input variables - age, gender, education & other personality characteristics
#The "output" columns are drug use for specific drugs and the pattern of use(past 6 months, past week etc.)
data = pds.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data")

#Dropping all the columns with irrelevant drugs that our model doesn't evaluate
column_drop_list = [14,15,16,17,19,20,21,22,24,25,26,27,28, 30, 31]
data = data.drop(data.columns[column_drop_list], axis=1)

#Labels the data that was imported from the website
headerList = ["ID", "Age", "Gender", "Education", "Country", "Ethnicity", "NScore", "EScore", "OScore", "AScore", "CScore", "Impulsiveness", "Sensation Seeing", "Alcohol","Marijuana", "Non-Prescribed Opiod", "Nicotine"]
data.columns = headerList
data.drop(columns=["ID"], inplace=True)

#The original data uses a CDX (where x is an integar) system to characterize the last use of the substance
#Since we're looking to predict the odds for a teenager, anything between 2-10 years is classified as never used
#0 represents never used, 1 represents used
ConversionDictionary = {"CL0" :0, "CL1" :0, "CL2" :0, "CL3" :1, "CL4" :1, "CL5" :1, "CL6" :1}
 
#iterates through the dataframe and converts the CDX system into 0 and 1's
for x in range(len(data)):
    data['Alcohol'].iloc[x]= ConversionDictionary[data['Alcohol'].iloc[x]]
    data["Marijuana"].iloc[x]= ConversionDictionary[data["Marijuana"].iloc[x]]
    data["Non-Prescribed Opiod"].iloc[x]= ConversionDictionary[data["Non-Prescribed Opiod"].iloc[x]]
    data["Nicotine"].iloc[x]= ConversionDictionary[data["Nicotine"].iloc[x]]
    
#Seperates the data columns into an input variable/x variable and output variable/y variable
input_var = data[["Age", "Gender", "Education", "Country", "Ethnicity", "NScore", "EScore", "OScore", "AScore", "CScore", "Impulsiveness", "Sensation Seeing"]]
output_var = data[["Alcohol","Marijuana", "Non-Prescribed Opiod", "Nicotine"]]

#Splits the data into train and test data which is used for later evaluating the model
X_train, X_test, y_train, y_test = train_test_split(input_var, output_var, test_size = 0.2, random_state = 11)

#Creates a neural network model to predict the probability of substance use for specific substances
#Neural Network Structure, 12 inputs, Hidden Layer 8, softmax classification(4)
model = Sequential()
model.add(Dense(8, input_dim = 12, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(4, activation='softmax'))
model.summary()

model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics =['accuracy'])
basic = model.fit(X_train, y_train, epochs = 100)

#Evaluates the model against the test data, and prints out the accuracy
print(model.evaluate(X_test, y_test)[1])

#This function generates the attributes of one highschool student
def genStudent():
    #The values for the variables were generated based on how the data within the dataset was processed by the author
    #The dataset is organized based on age, -0.95197 correlates to ages 18-24 (highschool senior age (~18))
    age = -0.95197
    #0.48246 is female -0.48246 is male, choice randomly selects one of them
    gender = np.random.choice([0.48246, -0.48246])
    #Original dataset normalized the educational levels, ex: -2.5 is dropout of highschool before 16, while 1.98 is Doctoral Degree
    education = np.random.normal(0, 1, 1)[0]
    
    #Country is only the United States
    country = -0.57009
    
    #All of these values were determined by the author and then normalized, therefore we can randomly generate students
    #and the characteristics with a normal distribution, mean =0 and standard deviation=1
    ethnicity = np.random.normal(0,1,1)[0]
    NScore = np.random.normal(0,1,1)[0]
    EScore = np.random.normal(0,1,1)[0]
    OScore = np.random.normal(0,1,1)[0]
    AScore = np.random.normal(0,1,1)[0]
    CScore = np.random.normal(0,1,1)[0]
    Impulsiveness = np.random.normal(0,1,1)[0]
    SS = np.random.normal(0,1,1)[0]
    
    #returns the attributes that we created for one individual student
    return age, gender, education, country, ethnicity, NScore, EScore, OScore, AScore, CScore, Impulsiveness, SS


#Total number of students that will use the specific drug
Alcohol_total = 0
Marijuana_total = 0
Non_Prescribed_Opiod_total = 0
Nicotine_total = 0
student_num = 10000

for x in range(student_num): #iterates through the number of students that we give
    current_Student = np.array(list(genStudent())) #Converts the tuple returned into a numpy array for neural network
    current_Student = current_Student.reshape(1,12)  #Reshapes the data into neural network format
    current_prediction = model.predict(current_Student) #runs the random values through the neural network
    #The neural network outputs the probability of the given student using the four drugs
    #The probability is then added to the total probability, of each drug
    Alcohol_total = Alcohol_total + current_prediction[0][0]
    Marijuana_total = Marijuana_total + current_prediction[0][1]
    Non_Prescribed_Opiod_total = Non_Prescribed_Opiod_total + current_prediction[0][2]
    Nicotine_total = Nicotine_total + current_prediction[0][3]
    
#Outputs the total number of students and the number of students that are expected to use the drug
print("Of the " + str(student_num) + " simulation students generated " + str(Alcohol_total) + " have used Alcohol")
print("Of the " + str(student_num) + " simulation students generated " + str(Marijuana_total) + " have used Marijuana")
print("Of the " + str(student_num) + " simulation students generated " + str(Non_Prescribed_Opiod_total) + " have used Non-Prescribed Opiods")
print("Of the " + str(student_num) + " simulation students generated " + str(Nicotine_total) + " have used Nicotine")

