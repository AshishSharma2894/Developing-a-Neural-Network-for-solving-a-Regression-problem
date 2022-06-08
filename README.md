# Developing-a-Neural-Network-for-solving-a-Regression-problem

Write a python program to create and test a neural network for solving a regression problem. 
In your development process, you need to test multiple hyperparameter to obtain the most appropriate network for your problem. 
The hyperparameters you’ll have to analyse are listed below:
1. Learning Rate
2. Batch Size
3. Number of epochs
4. Activation functions
5. Optimizers
6. Loss function
7. Number of Hidden Layers
8. Number of nodes in the hidden layers


OBS_1: when testing one hyperparameter, needed to fix all the others to properly analyse the impact of each hyperparameter in the network.
Uses cross-validation to find your best network. To evaluate the impact of different hyperparameter.Also, plotted the loss function graph to observe how the loss of the network reduces or increases based on the values of a chosen hyperparameter.

however the essential steps your code should include are:
• Load the dataset
• Split and clean the dataset
• Load and train the neural network with different hyperparameters
• The many strategies used for testing multiple hyperparameters
• Plotting the results for the best network obtained


# Best Network with our Parameters:
1. {"[0.001, 20, 100, 'relu', 'RMSProp', 'huber_loss', 4, 64]": {'Testing Data': [{'val_loss': 0.05252441391348839},{'val_cosine_proximity': 0.8408512473106384}],'Training Data': [{'loss': 0.020933888852596283}

![image](https://user-images.githubusercontent.com/99655823/172707443-d5a6dcd7-9e79-4a1f-9eb7-e313bb97f003.png)

It can be clearly seen from the result that when the learning rate is kept “0.001” it helps the code to prevent itself from over-fitting and with Batch size to be “20” it clearly means with approx. 
750 rows per batch the model performs better considering to smaller batch size. Number Of Epochs in this case is “100” it explains us that the greater number of times the dataset has to go through the model the more refined it gets, 
but we need to also keep in account that if we increase epochs to be too large of a value it will overfit the model. 
Activation Function under the best result is “Relu” which clearly gives us the output positive if input is positive, further optimizer and 
loss function are “RMSProp” and “huber_loss” the RMSProp must have helped in getting better tuning of batch sizes as it helps in converting batches into smaller batches, 
with huber loss it helps more above Squared errors as it is less sensitive to the outliners which can be due to the variability in measurement. 
Further 4 Hidden layers and 64 Neurons clearly states that in our model that more the hidden layers and neurons the better the model has performed.
