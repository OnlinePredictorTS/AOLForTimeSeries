# Adaptive Online Time Series Prediction

# How to run this code

Install Python 3.8.1 and Numpy version 1.18.1

- clone this project

    ```git clone https://github.com/OnlinePredictor/AdaptiveOnlineTimeSeriesPrediction.git```
- to run the experiments later you need to install joblib 

    ```pip install joblib```

# Overview

This repository contains different python files. 

These files define our problems we used to test our algorithm.
1. [RevisionExperimentSetting1.py](code/problems/RevisionExperimentSetting1.py)
2. [RevisionExperimentSetting2.py](code/problems/RevisionExperimentSetting2.py)
3. [RevisionExperimentSetting3.py](code/problems/RevisionExperimentSetting3.py)
4. [RevisionExperimentSettingReal1.py](code/problems/RevisionExperimentSettingReal1.py)
5. [RevisionExperimentSettingReal2.py](code/problems/RevisionExperimentSettingReal2.py)
6. [RevisionExperimentSettingReal3.py](code/problems/RevisionExperimentSettingReal3.py)

The following files contain the predictors.
1. [ArimaAutoregressor.py](code/predictors/ArimaAutoregressor.py)
2. [Jamil_algo1.py](code/predictors/Jamil_algo1.py)
3. [ARIMA_AO_Hedge.py](code/predictors/ARIMA_AO_Hedge.py)

Optimizers:
1. [RealOGD.py](code/optimizers/RealOGD.py)
2. [RealONS.py](code/optimizers/RealONS.py)
3. [ADAFtrl.py](code/optimizers/ADAFtrl.py)
4. [Ftrl4.py](code/optimizers/Ftrl4.py)


# Run experiments

To reproduce the results in our paper run [RunExperiments.py](code/RunExperiments.py).
```python PaperExperiments.py```
Running the experiments like this will start many threads and require a lot of computational power. If you do not want to run all experiments adjust the tasklist array in the main method.