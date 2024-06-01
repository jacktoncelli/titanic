import pandas as pd
import os
import datetime

from perceptron import TitanicPerceptron

# getting the data from the cleaned files
data_raw_dir = os.path.join(os.getcwd(), "data/clean")

cleaned_test_csv = os.path.join(data_raw_dir, "cleaned_test.csv")
cleaned_train_csv = os.path.join(data_raw_dir, "cleaned_train.csv")

train_df = pd.read_csv(cleaned_train_csv)
test_df = pd.read_csv(cleaned_test_csv)

# splitting the data into input and output
train_input_features = train_df.iloc[:, :-1]
train_output = train_df.iloc[:, -1]

# Parameters
num_features = train_input_features.shape[1]
epochs = 500
learning_rate = 0.003
l1_lambda = 0.01
l2_lambda = 0.01

# initializing the perceptron model
titanic_perceptron = TitanicPerceptron(num_features)

# training the model 
titanic_perceptron.train(input=train_input_features, target_output=train_output, 
                         num_epochs=epochs, learning_rate=learning_rate, 
                         l1_lambda=l1_lambda)

# load the best model state
titanic_perceptron.load_best_model()

# predict the output of the test data
test_predictions = titanic_perceptron.forward(test_df)
test_predictions_df = pd.DataFrame({'PassengerID' : range(892, 892 + len(test_predictions)),'Survived': test_predictions.flatten()})

# save predictions to a csv
curr_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")

predictions_dir = os.path.join(os.getcwd(), "predictions")
test_predictions_df.to_csv(os.path.join(predictions_dir, curr_time_str + 'predictions.csv'), index=False)