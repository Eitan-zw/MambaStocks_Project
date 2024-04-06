# Mamba Stock Prediction

Welcome to our Mamba Stock Prediction project! Here, we harness the power of the Mamba architecture to forecast stock prices accurately. By analyzing historical market data and employing innovative techniques, we demonstrate the effectiveness of Mamba in predicting future stock behavior. Explore our project as we delve into the realm of stock market forecasting with Mamba :)

![Image Description](https://github.com/Eitan-zw/MambaStocks_Project/blob/master/_11520c05-2fec-40dc-9263-98494b359cb9.jpeg)


## Data Processing

Our data processing begins with the `stockz.py` script, a versatile tool for fetching historical stock data from Yahoo Finance. This script extracts vital metrics like opening and closing prices, high and low prices, adjusted close, and trading volume.

Next, we delve into feature engineering, enriching our dataset with calculated metrics such as daily change, percentage change, and total trading amount. This process enhances our model's understanding of stock price dynamics, bolstering its predictive capabilities.

During model training, we meticulously select the most relevant features while removing any redundant ones. Furthermore, we rigorously split our data into training and testing sets to evaluate the model's performance on unseen data, ensuring its ability to generalize effectively beyond the training phase.

## Architecture

Our stock prediction model is built upon the Mamba framework, specifically leveraging the lightweight implementation available in [mamba-minimal](https://github.com/johnma2006/mamba-minimal). This simplified version encapsulates the essence of Mamba's state space modeling in a single PyTorch file.

![Mamba](https://github.com/state-spaces/mamba/blob/main/assets/selection.png)
> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**\
> Albert Gu*, Tri Dao*\
> Paper: https://arxiv.org/abs/2312.00752

### Model Overview

In our implementation, we adapted the Mamba model for stock prediction by integrating it with our custom features and requirements. One notable modification was the removal of the vocabulary embedding layer, aligning the model's architecture with our numerical data processing pipeline.

### Key Components

The Mamba-based model comprises essential components such as residual blocks, selective scanning algorithms, and input-dependent parameter modulation. These elements collectively enable the model to capture intricate patterns in stock price data while ensuring computational efficiency and predictive accuracy.

For further details on the Mamba framework and its implementation, we recommend referring to the original research papers and the annotated code available in the [Mamba](https://github.com/state-spaces/mamba) repository.

## Requirements

The code has been tested running under Python 3.12, with the following packages and their dependencies installed:
```
numpy==1.26.4
matplotlib==3.8.3
scikit-learn==1.4.1.post1
pandas==2.2.1
pytorch==1.7.1
```
## Results

### Example Scenario: Apple Stock

In an example scenario with Apple stock, historical data from 1986 to 2024 is downloaded and processed. The model is trained using MSE loss optimization and RMSprop as the optimizer.

Training the model on AAPL stock produced the results shown in Figures 1 and 2. Figure 1 illustrates the AAPL stock prediction result, while Figure 2 displays the loss during training.

![AAPL Stock Prediction Result](https://github.com/Eitan-zw/MambaStocks_Project/blob/master/apple%20stock%20n%3D100.jpg)
                                *Figure 1: AAPL Stock Prediction Result*

![Loss When Training - AAPL](https://github.com/Eitan-zw/MambaStocks_Project/blob/master/epoch%20vs%20loss%2040%20epoch.jpg)
                                *Figure 2: Loss When Training - AAPL*

### Cross-Stock Prediction

To assess the model's generalization capability, we tested the trained model on predicting another stock, NKE (Nike). Figure 3 depicts the prediction results using the AAPL-trained model on NKE stock data.

![NKE Prediction Using AAPL Trained Model](https://github.com/Eitan-zw/MambaStocks_Project/blob/master/nke%20stock%20n%3D100.jpg)
                                *Figure 3: NKE Prediction Using AAPL Trained Model*

## Usage

```
stockz.py -> run and input the required stock data
training.py -> change the parsing to allow for your desired stock data to be trained on, for example (aapl_stock_data).the model will be save in model.pt.
testing.py -> run with the relevant model.pth + stock_data file
```
## Further Work

### Trading Bot Integration
Consider integrating a trading bot to automate trading decisions based on the predicted stock prices. One such trading bot project is available [here](https://github.com/roeeben/Stock-Price-Prediction-With-a-Bot). Integrating the trained model with a trading bot could potentially enable automated trading strategies based on the predicted stock prices.

### Prediction for Future Trading Days
Extend the functionality of the prediction model to forecast stock prices for future trading days. This enhancement would involve modifying the code to accommodate predictions beyond the testing period. Implementing this feature would enable users to make informed decisions about potential future stock price movements, even in the absence of typical feature information.


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
