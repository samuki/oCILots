# oCILots

<p align="left">
  <img src="notebooks/ocilot.png" width="350" alt="">
</p>



## Collecting additional training data

The python script collect_data.py in the additional-data repository has been used to collect additional images from google maps The code in the script is an adapted version from the following phyton script available on github https://github.com/ardaduz/cil-road-segmentation/tree/master/additional-data. 

To run the script you need to add a google maps api key in the  placeholder section "PLACE YOUR GOOGLE API KEY HERE" in collect_data.py. 
To run the code you need to run the following commands. 

***REMOVED***
cd additional-data
python3 collect_data.py
***REMOVED***
The script is fetching data from cities defined in input_cities.csv. The file can also be found in the additional-data/ folder. 


***REMOVED*** 

## Data setup for training
To run the scripts defined in the next section and reproduce our results you need to have the same training data available + validation/test split. ***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

## Reproduce our results
Please make sure that before running the code you install all relevant python requirements. They can be found in our [requirements.txt](requirements.txt)


We have defined a wide set of configuration files which can be found in the folder [configs/](configs/). 

**Please note that some of the code needs to be run with a GPU.**

Reproducing the results with the configuration files can be done by passing the relevant configuration file to the main.py. You can for example reproduce the Segformer with augmentation data test with the following command. 
***REMOVED***
python3 main.py --config configs/segformer_augmentation.yaml
***REMOVED***
The trained mode, a submission csv and a log file will then be stored in a new folder in the directory [results](results/). The folder name is the timestamp when you started the experiment. 


**Please note:**
*Some of the experiments require you to load a pretrained model. For example the fine tune experiments are done by first training the model on the extra data. If you want to reproduce the entire experiment you need to change the checkpoint directory in the configuration file to point to the model trained on the extra data. Which experiments are affected is shown in the table below. Moreover, we added a comment to the relevant line in te configuration file*. 