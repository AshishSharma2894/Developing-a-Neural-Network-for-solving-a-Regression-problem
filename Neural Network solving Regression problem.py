
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
import pprint
import tensorflow as tf

"""
The Purpose of this code is to solve with the help of regression and to test the result with
multiple different parameters to understand the pricing of the area and better prediction
"""

# All the hyperparameters (8)are defined below with this Dictionary :
Main_Parameters={
    'Learning_Rate':[0.1,0.01,0.001],
    'Batch_Size':[10,15,20],
    'Number_Of_Epochs':[20,60,100],
    'Activation_Function':[None,"sigmoid",'relu'],
    'Optimizer':["adam","SGD","RMSProp"],
    'Loss_Function':["mean_squared_error","mean_absolute_error","huber_loss"],
    'Number_Of_Hidden_Layers':[2,3,4],
    'Number_Of_Nodes_In_Hidden_Layers':[64,32,16]}


def Avg_Value_For_Columns(columns):
    """It helps in providing the average value for the columns which are non zeros

    Args:
        columns : It holds the values of the columns in the dataset which is a series and is then further converted into dataframes

    Returns:
        The mean value of the column exluding all zeros 
    """
    data=columns.to_frame()
    return data.loc[data[data.columns[0]] !=0].mean()[0]

KCHouseData=pd.read_csv("/content/kc_house_data.csv")

for column in KCHouseData.iloc[:,np.r_[3:7,9:21]]:
    KCHouseData[column]=KCHouseData[column].replace(0,Avg_Value_For_Columns(KCHouseData.iloc[:,np.r_[3:7,9:21]][column]))

KCHouseData=KCHouseData.sample(frac=1)
Features=KCHouseData.iloc[:,3:21]
Target_Columns=np.array(KCHouseData.iloc[:,2])
Target_Columns=Target_Columns.reshape(-1,1)

"""
Here we are normalizing the data as all the values or parameters are in different scaler values.
"""
Input_Scaler=StandardScaler()
Output_Scaler=StandardScaler()
X_Scaler=Input_Scaler.fit(Features)
Y_Scaler=Output_Scaler.fit(Target_Columns)
Features=X_Scaler.transform(Features)
Target_Columns=Y_Scaler.transform(Target_Columns)

# Here we are splitting the data into training and testing in a ratio of 80:20
train_x,test_x,train_y,test_y=train_test_split(Features,Target_Columns,test_size=0.2,shuffle=True)

def Parameter_Filtering(Main_Parameters):
    """The purpose of this function is to generate the combinations of all the hyperparameters
    so that we can further work along to fetch the desired results.

    Args:
        Main_Parameters (dict):It holds the values that we have defined for each hyperparameters

    Returns:
        list:It provides list of all possible combinations
    """
    Possibilites=[]
    for i in range([len(i) for i in Main_Parameters.values()][0]):
        Set_Location=[]  
        for h in Main_Parameters:
          Set_Location.append(Main_Parameters[h][i])
        Possibilites.append(Set_Location)
    return Possibilites

Possibilites=Parameter_Filtering(Main_Parameters)
Prediction={}
for iters in Possibilites:
    for i in range(len(iters)):
        Prediction.update({(iters[i],i):[[iters[j] for j in range(len(iters)) if j!=i]for iters in Possibilites]})

Final_Parameters=[]

for m in Prediction:
    todo=m
    for options in Prediction[m]:
        options.insert(todo[1],todo[0])

    if options not in Final_Parameters:
        Final_Parameters.append(options)


Main_Result=[]

def Model_Created(parameters):
    """The Function here does the majority of the task as it trains the model and all the parameters are placed into the right spot to work around
    the model takes the hyperparameters as the input and represent with different hyperparameters.
    And further plotting of Loss Function and Metric which helps in visualisation of the data.

    Args:
        parameters (list): This holds all the possible hyperparameters for the model to train with and share the desired result
    """

    Model=Sequential()

    Model.add(Dense(units=parameters[7],input_dim=18,kernel_initializer='normal',activation=parameters[3]))

    for element in range(parameters[6]):
        Model.add(Dense(units=parameters[7],kernel_initializer='normal',activation=parameters[3]))

    Model.add(Dense(1,kernel_initializer='normal'))

    if parameters[4]=='adam':
        optimiser=tf.keras.optimizers.Adam(learning_rate=parameters[0])
    elif parameters[4]=='SGD':
        optimiser=tf.keras.optimizers.SGD(learning_rate=parameters[0])
    elif parameters[4]=='RMSProp':
        optimiser=tf.keras.optimizers.RMSprop(learning_rate=parameters[0])

    Model.compile(loss=parameters[5],optimizer=optimiser,metrics=['cosine_proximity'])

    Final_Model=Model.fit(train_x,train_y,validation_data=(test_x,test_y),batch_size=parameters[1],epochs=parameters[2])

    Main_Result.append(Final_Model.history)

    plt.plot(Final_Model.history['cosine_proximity'],label='cosine_Value_Training')
    plt.plot(Final_Model.history['loss'],label='loss_Value_Training')
    plt.xlabel('Number_Of_Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

    plt.plot(Final_Model.history['cosine_proximity'],label='Value_Cosine_Test')
    plt.plot(Final_Model.history['loss'],label='Value_Loss_Test')
    plt.xlabel('Number_Of_Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


def Main_Score(parameters):
    """This function performs the task of taking the data from the "def Model_Created(parameters)" which can be further used to calculate the score

    Args:
        parameters (list): This holds all the possible hyperparameters for the model to train with and share the desired result

    Returns:
        Main_Resultss(list): It returns us with the list of all values which are to be worked upon with.
    """
    Main_Resultss=[]

    for k in parameters:
        parameters_result={}
        Main_Result.clear()
        Model_Created(k)
        Train=[]
        test=[]
        Combined_Dataset={}
        for d in Main_Result[0]:
            if d =='loss' or d=='cosine_proximity' or d=='test':
                Train.append({d:Main_Result[0][d][-1]})
                Combined_Dataset.update({"Training Data": Train})
            else:
                test.append({d:Main_Result[0][d][-1]})
                Combined_Dataset.update({"Testing Data": test})
        
        parameters_result.update({str(k):Combined_Dataset})
        Main_Resultss.append(parameters_result)
    prints=pprint.PrettyPrinter(width=25,compact=True)

    return prints.pprint(Main_Resultss)

#Run below command to run Complete Dataset        
#Main_Score(Final_Parameters)

#Run below command to check for Best Three Parameters for the data set
#Main_Score([[0.001, 20, 100, 'relu', 'adam', 'huber_loss', 4, 16],[0.001, 20, 100, 'relu', 'RMSProp', 'huber_loss', 4, 32],[0.001, 20, 100, 'relu', 'RMSProp', 'huber_loss', 4, 64]])

