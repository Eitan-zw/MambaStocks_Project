import torch.nn as nn
from model import Mamba, ModelArgs
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error



parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False, help='Enable CUDA training.')
parser.add_argument('--ts-code', type=str, default='nke_stock_data', help='Stock code.')
parser.add_argument('--hidden', type=int, default=2, help='Dimension of representations.')
parser.add_argument('--layer', type=int, default=8, help='Number of layers.')
parser.add_argument('--n-test', type=int, default=300, help='Size of test set.')
parser.add_argument('--csv-path', type=str, default='dimensions.csv', help='Path to load dimensions.')
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

class Network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Args = ModelArgs(d_model=args.hidden, n_layers=args.layer)
        self.Mamba = nn.Sequential(
            nn.Linear(in_dim, args.hidden),
            Mamba(self.Args),
            nn.Linear(args.hidden, out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.Mamba(x)
        return x.flatten()

def load_model(path, in_dim, out_dim):
    model = Network(in_dim=in_dim, out_dim=out_dim)
    model.load_state_dict(torch.load(path))
    if args.cuda:
        model = model.cuda()
    model.eval()
    return model

def Predict(model, X_testing, close, n_test):
    x_test = torch.from_numpy(X_testing).float().unsqueeze(0)

    if args.cuda:
        x_test = x_test.cuda()

    mat = model(x_test)
    if args.cuda:
        mat = mat.cpu()
    val = mat.detach().numpy().flatten()

    # Extract actual closing prices for the testing period
    closed = close[-n_test:]

    # Generate predicted stock prices
    finalpredicted_stock_price = []
    pred = close[-n_test - 1]
    for i in range(n_test):
        pred = close[-n_test - 1 + i] * (1 + val[i])
        finalpredicted_stock_price.append(pred)

    return finalpredicted_stock_price, closed

# Read dimensions from CSV file
dimensions = pd.read_csv(args.csv_path, index_col=0)

# Read the CSV file
data = pd.read_csv(args.ts_code + '.csv')

# Drop unwanted columns
data.drop(columns=['Change'], inplace=True)

# Extract necessary columns
close = data.pop('Close').values
rate = 0.01 * data.pop('Percentage Change').values

# Extract features
X = data.iloc[:, 3:].values

# Split data into training and testing sets
X_testing = X[-args.n_test:, :]

# Load the trained model
model = load_model('trained_model.pth', in_dim=dimensions.loc['in_dim', 'Value'], out_dim=dimensions.loc['out_dim', 'Value'])

# Predict using the trained model
predictions, closed = Predict(model, X_testing, close, args.n_test)

# Convert date to datetime
time = pd.to_datetime(data['Date'][-args.n_test:])

# Plot showing only the testing period and predicted vs actual stock prices
plt.figure(figsize=(12, 7))
plt.plot(time, closed, color='green', label='Actual Stock Price (Testing Period)')
plt.plot(time, predictions, color='red', label='Predicted Stock Price (Testing Period)')
plt.title('Stock Price Prediction - Actual vs Predicted (Testing Period)')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Close', fontsize=12)
plt.legend()
plt.show()

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(closed, predictions)

# Print MSE
print("Mean Squared Error (MSE):", mse)


print("Predictions:")
for i, prediction in enumerate(predictions):
    print(f"Date {time.iloc[i].date()}: Prediction price is {prediction} and actual price is {closed[i]}")

