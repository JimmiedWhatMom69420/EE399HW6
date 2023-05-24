# EE399 HW-6

``Author: Marcel Ramirez``

``Publish Date: 5/15/2023``

``Course: Spring 2023 EE399``

##### This assignment focuses on training neural networks to predict what may happen in the future utilizing Lorenz equations for the Lorenz system.

This repository contains the code for the paper "Sensing with shallow recurrent decoder networks" by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz. SHallow REcurrent Decoders (SHRED) are models that learn a mapping from trajectories of sensor measurements to a high-dimensional, spatio-temporal state. For an example use of SHRED, see the iPython notebook example.ipynb.

The datasets considered in the paper "Sensing with shallow recurrent decoder networks" consist of sea-surface temperature (SST), a forced turbulent flow, and atmospheric ozone concentration. Cloning this repo will download the SST data used. Details for accessing the other datasets can be found in the supplement to the paper. 

# Summary

This assignment focuses on utilizing machine learning to analyze various factors affecting time series forecasting. Specifically, the project employs an LSTM/decoder model to predict sea-surface temperature. The objectives of this assignment include the following:

1. Downloading the example code and data for sea-surface temperature, which utilizes an LSTM/decoder model.
2. Training the model and generating visualizations of the results.
3. Analyzing the performance of the model in relation to the time lag variable.
4. Analyzing the performance of the model when subjected to different levels of noise by adding Gaussian noise to the data.
5. Analyzing the performance of the model based on the number of sensors.

In this project, we will explore the SHRED (Spatio-Historical Regularized Encoder-Decoder) model, a deep learning architecture designed to effectively handle complex temporal and spatial dependencies present in time series data. By combining encoder and decoder modules with regularization techniques, the SHRED model is capable of reconstructing states and forecasting sensors in multi-sensor time series datasets.

The key objectives of this project are as follows:

1. Implementation of the SHRED model for time series forecasting.
2. Preprocessing and preparation of time series data, including handling missing values, scaling, and generating input sequences.
3. Training the SHRED model on training data and optimizing the model parameters using validation data.
4. Evaluating the performance of the model on test data, including measuring reconstruction accuracy and forecasting error.
5. Visualizing the results, such as plotting ground truth and reconstructed time series data, as well as performance metrics.

To accomplish these goals, we will utilize Python and popular libraries such as NumPy, PyTorch, and Matplotlib. The project provides modular code for data preprocessing, model implementation, training, and evaluation, allowing users to apply the SHRED model to their own time series forecasting tasks.

Furthermore, the project includes a repository that contains example code and datasets. These resources serve to demonstrate the application of the SHRED model. Detailed instructions on installing dependencies, running the code, and interpreting the results can be found in the project's documentation.

Overall, this project serves as a valuable resource for researchers, data scientists, and practitioners who are interested in advanced time series forecasting techniques, particularly in scenarios involving spatially distributed sensor data.

## Theory
The Time Series Forecasting with SHRED Model project is built upon fundamental concepts and principles in the fields of time series analysis and deep learning. Understanding these theoretical foundations is essential for effectively utilizing and expanding the capabilities of the SHRED model. Presented below is a concise overview of the key theoretical concepts behind the project:
  $$H(\{y_{i}\}^{t}_{i=t-k}) = F(G(\{y_{i}\}^{t}_{i=t-k});W_{RN});W_{SD})$$

Time Series Forecasting
Time series forecasting involves predicting future values of a variable based on its past observations. This technique is widely employed in various domains to facilitate informed decision-making and planning. Analyzing time series data entails comprehending and modeling its temporal dependencies, trends, seasonality, and other patterns.

Deep Learning
Deep learning is a subset of machine learning that focuses on developing models capable of learning and representing complex data using artificial neural networks. Deep learning models, often composed of multiple layers, can automatically learn hierarchical representations from the input data.

Recurrent Neural Networks (RNNs)
Recurrent Neural Networks (RNNs) are a class of neural networks designed specifically for processing sequential data. RNNs possess recurrent connections that enable them to retain information across different time steps, thereby capturing temporal dependencies within time series data. However, traditional RNNs encounter challenges such as vanishing or exploding gradients and struggle to capture long-term dependencies effectively.

Encoder-Decoder Architectures
Encoder-Decoder architectures, also known as sequence-to-sequence models, find extensive application in various tasks such as machine translation and text generation. These architectures consist of two primary components: an encoder that processes the input sequence and captures its context, and a decoder that generates the output sequence based on the encoded representation. This framework has been adapted for time series forecasting, where the encoder-decoder structure captures temporal patterns and learns to generate future values.

SHRED Model
The SHRED (Spatio-Historical Regularized Encoder-Decoder) model combines the strengths of recurrent neural networks, encoder-decoder architectures, and regularization techniques for time series forecasting. It incorporates spatial information by selecting relevant sensor locations and learning their dependencies over time. By leveraging encoder-decoder modules, the model captures temporal patterns and generates accurate forecasts. Moreover, regularization techniques such as L1 and L2 regularization are employed to prevent overfitting and enhance generalization.

Through the integration of these theoretical concepts, the SHRED model effectively models complex dependencies within time series data, leading to accurate forecasts. The implementation of this project provides a practical and accessible framework for applying the SHRED model to a diverse range of time series forecasting tasks.

# Implementation

The SST data analysis using the LSTM/decoder algorithm involved a series of crucial steps. The algorithm's development and implementation process, accompanied by the provided code snippets, can be summarized as follows:

Data Acquisition and Normalization: Initially, the SST data was obtained by employing the load_data function, and subsequently, the MinMaxScaler from scikit-learn was utilized to normalize the data. The scaled data, stored in the transformed_X variable, served as the algorithm's input.

Model Architecture: The implementation incorporated the SHRED model, which comprises both an encoder and a decoder. The encoder component employed LSTM layers to process the input sequence and capture temporal information. Similarly, the decoder component employed LSTM layers to reconstruct or predict SST data based on the acquired context.

Training and Validation: Training and validation datasets were employed to train and validate the SHRED model. The training data and corresponding ground truth values were converted into PyTorch tensors (train_data_in, train_data_out) and used to generate the train_dataset. Similarly, the validation data was transformed and utilized to create the valid_dataset. The model was trained using the fit function, which optimized the model parameters based on the training data while monitoring the validation error to prevent overfitting.

Performance Evaluation: To assess the performance of the trained SHRED model, a test dataset was utilized. The test data was converted into PyTorch tensors (test_data_in, test_data_out) and employed to create the test_dataset. The model's predictions (test_recons) were obtained by passing the test input data through the SHRED model and subsequently inversely transformed using the inverse_transform method of the MinMaxScaler. The performance metric for the algorithm was determined by calculating the reconstruction error, which represents the normalized difference between the predicted and ground truth values. The evaluation of the algorithm's performance demonstrated its ability to capture temporal dependencies accurately and reconstruct or predict SST values effectively, as evidenced by the evaluation results. By adhering to these steps, the LSTM/decoder algorithm was successfully implemented and applied to analyze SST data.

# Computational Results

The SHRED model used is from this Github [repository](https://github.com/Jan-Williams/pyshred) by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz. There are some modifications to the original repository in order to add some visualization to the output and loading the full SST data based on a [fork](https://github.com/shervinsahba/pyshred) by Shervinsahba. The summary of the SHRED network is shown in the figure below.

```C
SHRED(
(Istm): LSTM(3, 64, num_1ayers=2, batch_first=True)
(linearl): Linear(in_features=64, out_features=3S8, bias-True)
(linear2): Linear(in_features=>64, out_features=>3S8, bias-True)
(linear3): Linear(in_features=>64, out_features=>3S8, bias-True)
( dropout) : Dropout(p=0.1, inp1ace=FaIse)
```


### Question 1 &2  Question 2

I must first split the data into adequate training and testing datasets. This will help train the model but, I have to fetch the data needed with the hyperparameters set for the function:

```python
num_sensors = 3 
lags = 52
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

# Splitting the data set

train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```

```python
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
for j in range(100):
    nn_input[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, :-1, :]
    nn_output[j * (len(t) - 1):(j + 1) * (len(t) - 1), :] = x_t[j, 1:, :]
    x, y, z = x_t[j, :, :].T
    ax.plot(x, y, z, linewidth=1)
    ax.scatter(x0[j, 0], x0[j, 1], x0[j, 2], color='r')
ax.view_init(18, -113)
plt.show()

```

Once we have the above set up, we can now generate input sequences to a SHRED model. The way we do this is by manipulating the sensor measurements such as reshaping or slicing. This will help enhance the training and test datasets to assign the valid input and outputs in their sequences. TimeSeriesDataset objects are instantiated by these data sets, and are utilized for the SHRED model for later steps in training and evaluation.

```d
### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for rec onstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```

To train a neural network (NN) to advance the solution from time t to t + ∆t for different values of ρ and evaluate its performance for future state prediction, you need to make the following modifications to the given code:

1.  Update the value of ρ to the desired values (e.g., 10, 28, and 40) by modifying the `rho` variable in the code.
    
2.  Modify the code to split the dataset into training and testing sets. Currently, the entire dataset is used for training. You can use the `train_test_split` function from scikit-learn to split the dataset into training and testing sets.
    
3.  Update the model architecture and training parameters accordingly.

Now we train the script on a neural network to advance the solution from ``t to t + Δt``  for ρ = 10, 28, 40

![image](https://media.discordapp.net/attachments/823976012203163649/1108644652447449088/FNq9eABJNdAAAAAElFTkSuQmCC.png?width=1438&height=443)
Next, lets see the computational output for ρ = 17 and 35 

![5eGgV0GDP00AAAAASUVORK5CYII.png (916×426) (discordapp.net)](https://media.discordapp.net/attachments/823976012203163649/1108644651377905684/5eGgV0GDP00AAAAASUVORK5CYII.png?width=1296&height=603)


### Question 2

To compare feed-forward, LSTM, RNN and Echo State Netowrks for forecasting the dynamics of the Lorenz system, we must briefly follow these:

1.  The Lorenz equations and system parameters are defined.
    
2.  Initial conditions are generated for the Lorenz system.
    
3.  Input and output arrays are prepared for the neural networks by integrating the Lorenz equations.
    
4.  The data is split into training and testing sets.
    
5.  Feed-Forward Neural Network (FNN):
    
    -   A feed-forward neural network model is created using the Keras Sequential API.
    -   The model is trained on the training data using the mean squared error loss.
    -   The model makes predictions on the testing data.
6.  LSTM:
    
    -   An LSTM model is created using the Keras Sequential API.
    -   The model is trained on the training data using the mean squared error loss.
    -   The model makes predictions on the testing data.
7.  RNN:
    
    -   An RNN model is created using the Keras Sequential API.
    -   The model is trained on the training data using the mean squared error loss.
    -   The model makes predictions on the testing data.
8.  Echo State Network (ESN):
    
    -   An Echo State Network model is created using the `reservoirpy` library.
    -   The model is trained on the training data.
    -   The model makes predictions on the testing data.
9.  The predictions from each model are compared using mean squared error (MSE) as the evaluation metric.
    

The code allows for a comparison of the performance of different neural network models in forecasting the dynamics of the Lorenz system.

Below is the computational output for it. Running all models giving the mean square errors for rho of 10 and 35

---------------------
| Mean Squared Error | Rho |
| -- | -- |
| 0.15224202827965347 | 10 |
| 2331.242977221156 | |
| 10145.713553004734 ||
|1.7928471466364329 | 35 |
| 2290.874895267006 ||
|9785.374156340838 ||

Below is the code for the models in python:

```python
# Feed-Forward Neural Network (FNN) 
model_ff = Sequential() 
model_ff.add(Dense(32, activation='relu', input_shape=(3,))) model_ff.add(Dense(32, activation='relu')) model_ff.add(Dense(3)) model_ff.compile(optimizer='adam', loss='mse') model_ff.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0) Y_pred_ff = model_ff.predict(X_test) 

# LSTM 
model_lstm = Sequential() model_lstm.add(LSTM(32, activation='relu', input_shape=(1, 3))) 
model_lstm.add(Dense(3)) 
model_lstm.compile(optimizer='adam', loss='mse') model_lstm.fit(X_train[:, np.newaxis, :], Y_train, epochs=50, batch_size=32, verbose=0) Y_pred_lstm = model_lstm.predict(X_test[:, np.newaxis, :]) 
# RNN 
model_rnn = Sequential() 
model_rnn.add(SimpleRNN(32, activation='relu', input_shape=(1, 3))) model_rnn.add(Dense(3)) model_rnn.compile(optimizer='adam', loss='mse') 
model_rnn.fit(X_train[:, np.newaxis, :], Y_train, epochs=50, batch_size=32, verbose=0) Y_pred_rnn = model_rnn.predict(X_test[:, np.newaxis, :]) 
# Echo State Network (ESN) 
model_esn = ESN(n_inputs=3, n_outputs=3, n_reservoir=1000) model_esn.fit(X_train, Y_train) Y_pred_esn = model_esn.predict(X_test)
```

Below is the computational output for this code: 

```python
Running Model-1:   0%|          | 0/1 [00:00<?, ?it/s]

Running Model-1:   0%|          | 0/1 [00:00<?, ?it/s]

Running Model-1: 732it [00:00, 7301.54it/s]          

…

Running Model-1: 90000it [00:12, 7178.04it/s]

Fitting node Ridge-0...

Running Model-1: 100%|██████████| 1/1 [00:12<00:00, 12.58s/it]

Running Model-1: 10000it [00:01, 8259.51it/s]

Mean squared error: 0.15224202827965347 for rho = 10

Running Model-1: 100000it [00:11, 8343.80it/s]

Mean squared error: 2331.242977221156

Running Model-1: 100000it [00:12, 7821.03it/s]

Mean squared error: 10145.713553004734  
_____

Running Model-1:   0%|          | 0/1 [00:00<?, ?it/s]

Running Model-1:   0%|          | 0/1 [00:00<?, ?it/s]

Running Model-1: 713it [00:00, 7107.06it/s]          

…

Running Model-1: 90000it [00:12, 7310.28it/s]

Fitting node Ridge-0...

Running Model-1: 100%|██████████| 1/1 [00:12<00:00, 12.35s/it]

Running Model-1: 10000it [00:01, 8144.66it/s]

Mean squared error: 1.7928471466364329 for rho = 35

Running Model-1: 100000it [00:12, 8259.11it/s]

Mean squared error: 2290.874895267006

Running Model-1: 100000it [00:12, 7923.76it/s]

Mean squared error: 9785.374156340838

```

Below is the computational output of this model, plotting the results:

![world forecast][https://github.com/JimmiedWhatMom69420/EE399HW6/blob/main/world.png]

### Question 3: time lag variable

The code below will help evaluate the SHRED's performance in many selected lag times and put them in an array. Next, we will run these tests with the lag times I choose:

```d
# Define the time lag values to evaluate
time_lag_values = [10, 20, 30, 40, 50]

# Load the data and perform scaling
load_X = load_data('SST')
sc = MinMaxScaler()
sc = sc.fit(load_X)
transformed_X = sc.transform(load_X)

# Define the number of sensors and other parameters
num_sensors = 3
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

# Define empty list to store performance metrics for each time lag
performance_metrics = []

# Iterate over the time lag values
for lags in time_lag_values:
    # Generate input sequences to the SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i + lags, sensor_locations]

    # Generate training, validation, and test datasets for state reconstruction and sensor forecasting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Create and train the SHRED model
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)

    # Calculate reconstruction error
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    reconstruction_error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)

    # Store the performance metric for the current time lag value
    performance_metrics.append(reconstruction_error)
```

### Question 4

Next thing we want to look at is if we add noise to the data. The type of noise we will analyze is gaussian noise. Code shown below with the figure outputted:

```d
# Define the noise levels to evaluate
noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05]

# Load the data and perform scaling
load_X = load_data('SST')
sc = MinMaxScaler()
sc = sc.fit(load_X)
transformed_X = sc.transform(load_X)

# Define the number of sensors and other parameters
num_sensors = 3
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

# Define empty list to store performance metrics for each noise level
performance_metrics = []

# Iterate over the noise levels
for noise_level in noise_levels:
    # Generate noisy data by adding Gaussian noise to the scaled data
    noisy_X = transformed_X + np.random.normal(loc=0, scale=noise_level, size=transformed_X.shape)

    # Generate input sequences to the SHRED model
    all_data_in = np.zeros((n, 1, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = noisy_X[i, sensor_locations].reshape(1, -1)

    # Generate training, validation, and test datasets for state reconstruction and sensor forecasting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    train_data_out = torch.tensor(transformed_X[train_indices], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Create and train the SHRED model
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)

    # Calculate reconstruction error
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    reconstruction_error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)

    # Store the performance metric for the current noise level
    performance_metrics.append(reconstruction_error)

# Plot the performance as a function of the noise level
plt.plot(noise_levels, performance_metrics, marker='o')
plt.xlabel('Noise Level')
plt.ylabel('Reconstruction Error')
plt.title('Performance as a Function of Noise Level')
plt.show()
```

#### Computational Output for this Question

```d
epoch: 560 valid_error: tensor(0.2613, device='cuda:0'):  56%|█████▌    | 559/1000 [01:22<01:05,  6.77it/s]
epoch: 460 valid_error: tensor(0.2611, device='cuda:0'):  46%|████▌     | 459/1000 [01:08<01:20,  6.73it/s]
epoch: 480 valid_error: tensor(0.2654, device='cuda:0'):  48%|████▊     | 479/1000 [01:10<01:17,  6.75it/s]
epoch: 300 valid_error: tensor(0.2708, device='cuda:0'):  30%|██▉       | 299/1000 [00:44<01:44,  6.73it/s]
epoch: 300 valid_error: tensor(0.2697, device='cuda:0'):  30%|██▉       | 299/1000 [00:44<01:43,  6.75it/s]
```

![noise level](https://github.com/JimmiedWhatMom69420/EE399HW6/blob/main/NoiseLvl.png)

### Question 5

Final question, this time lets do an analysis of the performance as a function of the number of sensors

```d
# Define the number of sensors to evaluate
sensor_counts = [1, 2, 3, 4, 5]

# Load the data and perform scaling
load_X = load_data('SST')
sc = MinMaxScaler()
sc = sc.fit(load_X)
transformed_X = sc.transform(load_X)

# Define the time lag and other parameters
time_lag = 52
n = load_X.shape[0]
m = load_X.shape[1]

# Define empty list to store performance metrics for each number of sensors
performance_metrics = []

# Iterate over the number of sensors
for num_sensors in sensor_counts:
    # Select random sensor locations
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    # Generate input sequences to the SHRED model
    all_data_in = np.zeros((n - time_lag, time_lag, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+time_lag, sensor_locations]

    # Generate training, validation, and test datasets for state reconstruction and sensor forecasting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    train_data_out = torch.tensor(transformed_X[train_indices + time_lag - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + time_lag - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + time_lag - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # Create and train the SHRED model
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)

    # Calculate reconstruction error
    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    reconstruction_error = np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)

    # Store the performance metric for the current number of sensors
    performance_metrics.append(reconstruction_error)

# Plot the performance as a function of the number of sensors
plt.plot(sensor_counts, performance_metrics, marker='o')
plt.xlabel('Number of Sensors')
plt.ylabel('Reconstruction Error')
plt.title('Performance as a Function of Number of Sensors')
plt.show()
```
![](https://github.com/JimmiedWhatMom69420/EE399HW6/blob/main/numSensors.png)




