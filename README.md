# Team oCILots
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

# Instruction to run the project code

## Collecting additional training data

The python script collect\_data.py in  [additional-data/](additional-data) has been used to collect additional images from Google maps. The code in the script is an adapted version from the following phyton script available on GitHub https://github.com/ardaduz/cil-road-segmentation/tree/master/additional-data. 

To run the script you need to add a Google maps api key in the  placeholder section "PLACE YOUR GOOGLE API KEY HERE" in collect\_data.py. 
To run the code you need to run the following commands. 

***REMOVED***
cd additional-data
python3 collect_data.py
***REMOVED***
The script is fetching data from cities defined in input\_cities.csv. The file can also be found in the additional-data/ folder. 


***REMOVED*** 

## Data setup for training
To run the scripts defined in the next section and reproduce our results you need to have the same training data available + validation/test split. ***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

## Segmentation Library
Due to performance issues with most python graph libraries, the underlying code to perform the minimum-cut-based segmentation was implemented in C++ using the Boost Graph Library.
The corresponding code and Makefile is located in the cut-lib directory.
In order to build a shared object that can be called from the python wrapper, run `make` in this directory.
Note that building the code requires a C++-compiler with support for C++-20 and Boost for the Boost Graph Library (we used g++ 11.3.0 and Boost 1.79.0-1).
The segmentations can be invoked from other parts of the codebase using the `RBFLogSegmenter` class by instantiating it and calling either the `segment` method or the call-operator on the object, e.g. as
***REMOVED***python
segmenter = cut.RBFLogSegmenter(sigma=10., lambd=0.1, resolution=100)
# let images be a (batch_size, W, H) numpy-array of either np.float32 or np.float64
images = segmenter(images)
***REMOVED***
Similarly, the directional segmenter (that was only used in some initial experiments), can be used e.g. as
***REMOVED***python
segmenter = cut.DirectionSegmenter(lambda_pred=1, lambda_dir=2000, radius=20, delta_theta=math.pi / 16)
# let images be a (batch_size, W, H) numpy-array of np.int32
images = segmenter(images)
***REMOVED***
Note that the underlying library is only built for specific input types, namely 32- or 64-bit floating point for `RBFLogSegmenter` and 32-bit unsigned integers for `DirectionSegmenter`.

In order to run a search for segmentation parameters on a given set of predictions and groundtruths, copy both the predictions and the groundtruths into files `predictions.npy` and `groundtruths.npy` into a directory.
Then, set the key PREDICTIONS\_BASE\_DIR as the name of that directory in the config-file and run `cut.py`.


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

- Standard Training - configs/segformer\_standard.yaml
- Augmented - configs/segformer\_augementation.yaml
- Extra Data - configs/segformer\_extra\_data.yaml
- Fine-Tune - configs/segformer\_fine\_tune.yaml
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


