import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
import os
import datetime

# getting the data from the cleaned files
data_raw_dir = os.path.join(os.pardir, os.pardir, "data/clean")

cleaned_test_csv = os.path.join(data_raw_dir, "cleaned_test.csv")
cleaned_train_csv = os.path.join(data_raw_dir, "cleaned_train.csv")

train_df = pd.read_csv(cleaned_train_csv)
test_df = pd.read_csv(cleaned_test_csv)

# splitting the data into input and output
train_input_features = train_df.iloc[:, :-1]
train_output = train_df.iloc[:, -1]

# create decision tree
tree = DecisionTreeClassifier(criterion="log_loss", splitter="random", max_depth=5)

# train it on training data
tree = tree.fit(train_input_features, train_output)

# predict for the test data
predicted = tree.predict(test_df)
test_predictions_df = pd.DataFrame({'PassengerID' : range(892, 892 + len(predicted)),'Survived': predicted.flatten()})

# save predictions to a csv
curr_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")

predictions_dir = os.path.join(os.pardir, os.pardir, "predictions")
test_predictions_df.to_csv(os.path.join(predictions_dir, curr_time_str + 'predictions.csv'), index=False)
print(f"Predictions saved to {os.path.join(predictions_dir, curr_time_str + 'predictions.csv')}")