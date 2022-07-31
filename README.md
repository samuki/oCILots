# Team oCILots
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

# Instruction to run the project code

## Collecting additional training data

The python script collect_data.py in  [additional-data/](additional-data) has been used to collect additional images from Google maps. The code in the script is an adapted version from the following phyton script available on GitHub https://github.com/ardaduz/cil-road-segmentation/tree/master/additional-data. 

To run the script you need to add a Google maps api key in the  placeholder section "PLACE YOUR GOOGLE API KEY HERE" in collect_data.py. 
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
The trained model, a submission csv and a log file will then be stored in a new folder in the directory [results](results/). The folder name is the timestamp when you started the experiment. 


**Please note:**
*Some of the experiments require you to load a pretrained model. For example, the fine tune experiments are done by first training the model on the extra data. If you want to reproduce the entire experiment you need to change the checkpoint directory in the configuration file to point to the model trained on the extra data. Which experiments are affected is shown in the table below. Moreover, we added a comment to the relevant line in the configuration file*. 

### List of Experiments
The following list shows the different configuration files used to reproduce the experiments. Please note that the metrics shown in the report table are not the final training results but the best results obtained during training. For the outputs please see the logs.txt created during training. 


SegFormer:

- Standard Training - configs/segformer_standard.yaml
- Augmented - configs/segformer_augementation.yaml
- Extra Data - configs/segformer_extra_data.yaml
- Fine-Tune - configs/segformer_fine_tune.yaml
  - Please note that for the fine tuning case we first need to train the model on the extra data and then you need to **manually change*** the config file such that the correct model checkpoint is loaded. 


### Majority Voting
To run the results of our final submission you need to train the following models and then apply the [majority voting script](majority_voting.py) to the output data. *(For instructions on how to run the specific models please check the description above)*
1. SegFormer Augmented Data 
2. SegFormer Fine Tune
3. 
4. 

To run the majority voting script you need to provide the different csv files from the different model runs to the script. You can do that by passing the different files obtained during training to the script. The files can be found in the corresponding training folder in the results directory. The majority voting scripts create a majority_voting_result.csv file which is the final file we submitted to the Kaggle competition. 

Example of how to run the majority_voting.py script. 
***REMOVED***
python3 majority_voting.py -i unet_1.csv unet_2.csv segformer_augmented.csv segformer_finetuned.csv 
***REMOVED***


