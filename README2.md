

# MambaStockz: Selective state space model for stock prediction

Mamba is a deep learning architechure developed by Albert Gu and Tri Dao unvieled in 2023. The Mamba Stockz is a novel implementation of Mamba to the field of stock prediction. Taking a stock of your choosing, dividing the data to train and test and creating an accurate prediciton. 
paper: https://arxiv.org/abs/2312.00752

## Backround
This project introduces our mamba implementation for stock prediction, a stock prediction model based on the Mamba framework. we use historical market data without relying on handcrafted features or extensive preprossessing, offering great predictive performance. Empirical evaluations across multiple stocks demonstrate its accuracy. This work underscores the efficacy of the Mamba approach in time-series forecasting. 

## Files
```
stockz.py -> code to data mine financial data
training.py -> train the model on particular stock data with your desired argument changes
testing.py -> test the model trained on the stock of your choosing
model.py -> the model architechure
```
## Requirements

The code has been tested running under Python 3.12, with the following packages and their dependencies installed:
```
numpy==1.16.5
matplotlib==3.1.0
sklearn==0.21.3
pandas==0.25.1
pytorch==1.7.1
```

## Usage

```
stockz.py -> run and input the required stock data
training.py -> change the parsing to allow for your desired stock data to be trained on, for example (aapl_stock_data).the model will be save in model.pt.
testing.py -> run with the relevant model.pth + stock_data file
```

## Options
```
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
```

```
