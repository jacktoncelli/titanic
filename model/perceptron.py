import numpy as np

class TitanicPerceptron():
    def __init__(self, input_size):
        self.input_size = input_size
        
        # initialize the weights as column vector of all zeros as default
        self.weights = np.zeros((input_size, 1), dtype=np.float64)
        
        # initialize the bias to a row vecotr of all zeros
        self.bias = np.zeros((1, 1), dtype=np.float64)
        
        # variables to keep track of best model state so far
        self.highest_accuracy = 0
        self.best_weights = None
        self.best_bias = None
    
    """
    Makes binary predictions on the given input using the current state of the weights
    """
    def forward(self, input):
        # get the dot product of the input and weight vector, add the bias term
        net_input = np.dot(input, self.weights) + self.bias
        
        # classify all greater or equal to 0 as 1 and all less than 0 as 0
        predictions = np.where(net_input >= 0, 1, 0)
        return predictions
    
    """
    For the given input and target output, finds the errors in predictions
    """
    def backward(self, input, target_output):  
        predictions = self.forward(input=input)
        errors = target_output - predictions
        return errors
    
    """
    For the given input and target output, will update the weights of this perceptron for
    num_epochs epochs according to the passed learning rate
    """
    def train(self, input, target_output, num_epochs, learning_rate=1):
        # iterate num_epoch times to update the weights
        for e in range(num_epochs):
            # for each epoch, go through each sample of the outputs and update as needed
            for i in range(target_output.shape[0]):
                input_data = input.iloc[i, :].values.reshape(1, self.input_size)
                true_label = target_output[i]
                
                # Forward pass
                net_input = np.dot(input_data, self.weights) + self.bias
                prediction = np.where(net_input >= 0, 1, 0)
                
                # Check if prediction is incorrect
                if prediction != true_label:
                    # Compute error
                    error = true_label - prediction
                    
                    # Update weights and bias
                    self.weights += learning_rate * error * input_data.T
                    self.bias += learning_rate * error.reshape(-1, 1)
                
            accuracy = self.evaluate(input, target_output)
            if accuracy > self.highest_accuracy:
                self.highest_accuracy = accuracy
                self.best_weights = np.copy(self.weights)
                self.best_bias = np.copy(self.bias)

            print(f"After epoch #{e}, accuracy on input is: {accuracy:.4f}, highest accuracy so far: {self.highest_accuracy:.4f}")
    """
    For the given input and target outputs, returns the accuracy rate for correct predictions
    """
    def evaluate(self, input, target_output):
        predictions = self.forward(input).reshape(-1)
        accuracy = np.sum(predictions == target_output) / target_output.shape[0]
        return accuracy
    
    def load_best_model(self):
        if self.best_weights is not None and self.best_bias is not None:
            self.weights = np.copy(self.best_weights)
            self.bias = np.copy(self.best_bias)
            print("Best model loaded.")
        else:
            print("No best model found. Train the model first.")