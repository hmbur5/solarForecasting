# solarForecasting

SOLAR FORECASTING

The objective of this project was to create a model that can predict the future 24-hour solar power output of home solar panels around Australia using weather forecasts. This prediction is valuable as non-variable energy sources often are not easily dispatched (eg. coal power plants which can require 12 hours to warm up), and so generous warning time can aid their ability to supplement variable renewable energy sources. 

Our model was implemented using historic weather and corresponding solar power generation data, which we used to train and evaluate an LSTM. Overall, the model had an average absolute error of 0.063 kWatts

![results](https://github.com/hmbur5/solarForecasting/blob/main/results.png?raw=true)


This model was deployed using AWS, and a webpage was made to access this information easily such that users can identify locations where additional energy will be required.

[![demo](https://drive.google.com/uc?export=view&id=1tRKF9aDC9HqLUsv3JGMEbBbEYogiZ_Wu/preview](https://drive.google.com/uc?export=view&id=1tRKF9aDC9HqLUsv3JGMEbBbEYogiZ_Wu)

