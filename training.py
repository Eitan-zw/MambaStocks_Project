import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Mamba, ModelArgs
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False, help='Enable CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2, help='Dimension of representations.')
parser.add_argument('--layer', type=int, default=8, help='Number of layers.')
parser.add_argument('--n-test', type=int, default=300, help='Size of test set.')
parser.add_argument('--ts-code', type=str, default='aapl_stock_data', help='Stock code.')
parser.add_argument('--csv-path', type=str, default='dimensions.csv', help='Path to save dimensions.')
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

set_seed(args.seed, args.cuda)

class Network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Args = ModelArgs(d_model=args.hidden, n_layers=args.layer)
        self.Mamba = nn.Sequential(
            nn.Linear(in_dim, args.hidden),
            Mamba(self.Args),
            nn.Linear(args.hidden, out_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        x = self.Mamba(x)
        return x.flatten()

def train_model(X_training, Y_training):
    clf = Network(len(X_training[0]), 1) # Initialize the neural network model
    opt = torch.optim.RMSprop(clf.parameters(), lr=args.lr, weight_decay=args.wd)  # Define the optimizer

    # Convert data to PyTorch tensors
    x_train = torch.from_numpy(X_training).float().unsqueeze(0)
    y_train = torch.from_numpy(Y_training).float()

    # Move model and data to GPU if CUDA is available
    if args.cuda:
        clf = clf.cuda()
        x_train = x_train.cuda()
        y_train = y_train.cuda()

    # Training the model
    losses = []
    for epoch in range(args.epochs):
        clf.train()
        loss = F.mse_loss(clf(x_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            print('Epoch %d vs Loss: %.4f' % (epoch, loss.item()))
            losses.append(loss.item())
            if len(losses) > 0:
                plt.plot(range(len(losses)), losses)  # Plot epochs vs. loss
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Epoch vs Loss')
                plt.pause(0.01)  # Pause for a short time to update the plot

    # Save dimensions to CSV file
    dimensions = {'in_dim': len(X_training[0]), 'out_dim': 1}
    df = pd.DataFrame.from_dict(dimensions, orient='index', columns=['Value'])
    df.to_csv(args.csv_path)

    return clf


def save_model(model, path):
    torch.save(model.state_dict(), path)

# Read the CSV file
data = pd.read_csv(args.ts_code + '.csv')

# Drop unwanted columns
data.drop(columns=['Change'], inplace=True)

# Extract necessary columns
close = data.pop('Close').values
rate = 0.01 * data.pop('Percentage Change').values

# Extract features and target variables
X = data.iloc[:, 3:].values
Y = rate

# Split data into training and testing sets
X_training, Y_training = X[:-args.n_test, :], Y[:-args.n_test]

# Train the model
trained_model = train_model(X_training, Y_training)

# Save the trained model
save_model(trained_model, 'trained_model.pth')
