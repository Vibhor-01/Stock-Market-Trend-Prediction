# Stock-Market-Trend-Prediction
This is a Stock Market Trend Prediction Project Based on Deep Learning.

Here in this project, I have used LSTM deep learning model, and have used the paramter of 100 Days and 200 Days moving averages, to predict the stock trend. The web application of the project is designed using Streamlit web application.

The LSTM model has 4 layers, with the first layer being the input layer, and the last layer being the ouput layer, which produces the output. the trend of the stock prices is showcased using a plt plot, with the comparison of original price vs the predicted stock price. the predicted stock price is reflected in red colour while original stock price is reflected in blue colour. 

The dataset, is not downloaded from any Data science website, or dataset. It is read in real time, from yahoo finance website using yfinance package. This yfinance package automatically creates a Dataframe for analysis (the features of dataset are automatically put into the format and rows and columns, and are fully editable). The start and end date of the stock prices is set by the project creator, here in this project the dates are taken from 1st Jan,2010. 

A common practice used by the stock brokers, is the 100 or 200 days moving averages. In this approach the last 100 days stock market trend is observed and averaged and if the moving average prices are moving up numerically that means that the stock will be profitable in future, as its price shall increase. so, in this project I have calculated the last 100 and 200 moving average for day 1, then day 2, day 3 and so on and ploted it onto a plt graph for comparison of the trends. 

The model is trained on the Open day prices of the stock, i.e the price of the stock at the start of the day. The data set is divided into sets of traning and testing with 70% of the data being the Training and 30% being testing. The deep learning model contains 4 layers (the programmer can have as many layers as required, but will have to set the dropout accordingly). Each of the layers have 50, 60, 80 and 120 units respectively, with 0.2, 0.3, 0.4, 0.5 droupout rates. The last layer which is the output layer has only 1 unit with no droupout layer. 

To compile the model, I have used the 'adam' optimizer as it can perform stochastic gradient descent for more accurate error finding, and have used mean squared error to estimate the losses. the model is trained on 100 epochs and is then saved in a h5 format with the name 'stock_pred.h5'. Lastly the data is scaled using fit_transformed to attenuate it to a scale of 0-10. 
