# Mamba Stock Prediction

Welcome to our Mamba Stock Prediction project! Here, we harness the power of the Mamba architecture to forecast stock prices accurately. By analyzing historical market data and employing innovative techniques, we demonstrate the effectiveness of Mamba in predicting future stock behavior. Explore our project as we delve into the realm of stock market forecasting with Mamba :)

## Data Processing

Our data processing begins with the `stockz.py` script, a versatile tool for fetching historical stock data from Yahoo Finance. This script extracts vital metrics like opening and closing prices, high and low prices, adjusted close, and trading volume.

Next, we delve into feature engineering, enriching our dataset with calculated metrics such as daily change, percentage change, and total trading amount. This process enhances our model's understanding of stock price dynamics, bolstering its predictive capabilities.

During model training, we meticulously select the most relevant features while removing any redundant ones. Furthermore, we rigorously split our data into training and testing sets to evaluate the model's performance on unseen data, ensuring its ability to generalize effectively beyond the training phase.

## Architecture

Our stock prediction model is built upon the Mamba framework, specifically leveraging the lightweight implementation available in [mamba-minimal](https://github.com/johnma2006/mamba-minimal). This simplified version encapsulates the essence of Mamba's state space modeling in a single PyTorch file.

### Model Overview

In our implementation, we adapted the Mamba model for stock prediction by integrating it with our custom features and requirements. One notable modification was the removal of the vocabulary embedding layer, aligning the model's architecture with our numerical data processing pipeline.

### Key Components

The Mamba-based model comprises essential components such as residual blocks, selective scanning algorithms, and input-dependent parameter modulation. These elements collectively enable the model to capture intricate patterns in stock price data while ensuring computational efficiency and predictive accuracy.

For further details on the Mamba framework and its implementation, we recommend referring to the original research papers and the annotated code available in the [Mamba](https://github.com/state-spaces/mamba) repository.
