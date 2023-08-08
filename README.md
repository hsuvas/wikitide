# Weakly Supervised Classfier on wikipedia temporal definition pairs
### A reporsitory containing pairs of Temporal Definition dataset extracted from wikipedia articles and Classfier trained to detect information change in the definitions
--------------------------------------------------------------------------------------------------------------------------------
## TL;DR: Executing the code

To create the dataset

```
python3 data_prep.py

```

To run the classifier with bootstrapping(for multiple sets of models)

```

python3 LM_multiple_models.py 
    -i <input_dir>
    -o <output_dir>

```

## Description

We introduce a weakly supervised classifier to detect information change in wikipedia definitions. The key idea behind this is devise a classifier to detect wikipedia definitions change over time and train it using the concept of weak supervision. We consider the problem of detecting change as a three class classification problem, where each class represent the following.

- For a title (or term),given two definitions (def1 and def2) for two timespans (timespan 1 and timespan2)
    - Class 2 if def1 and def2 are different AND something fundamental happened to term or the knoweldge we had about term fundamentally changed between the dates timespan1 and timespan2.
    - Class 1 if def1 and def2 are different BUT the difference is mostly semantic or aesthetic, not about new or updated knowledge about term.
    - Class 0 if def1 and def2 are conveying basically the same information.
        
The process is as following.
1. Prepare a small labelled seed dataset (3000 definition pairs in this case) and train the classifier
2. Given an big unlabelled dataset, do
    a. Generate the labels for  the items in the dataset (using [![Temparature Scaling](https://github.com/gpleiss/temperature_scaling)])
    
    b. Consider top n pairs for the three labels and 
    
        i.  Put them in training data
        
        ii. Delete the n definition pairs from the unlabelled dataset
    
    c. Re-train the model
3. Continue step 2 until unlabelled dataset is empty.

## Data

The Dataset is created in two parts: labelled data and unlabelled data in a 30-70 split. We consider a dataset with 3000 rows of [(timestamp1,definition1),(timestamp2,definition2), label] format. We leverage GPT3.5 to find the labels of these dataset, and prompt GPT3.5 in four different ways to extract the labels and calculate the inter-annotation agreement, which is moderatly good. We then split the labelled dataset into Train and Test column. Since we don't do any finetuning over the data, we don't create the validation data split. We set the unlabelled dataset in the same format as of our labelled data, without the labels. Overall our dataset looks like the following

- Unlabelled Data
    - Number of rows: 9289 

- Labellled Data
    - Number of rows: 2999
    - Number of labels: 3
    - Data Splits:
        - Train Data: 1799 Rows
        - Test Data: 1200 Rows


## Results
Will be put down here once its done.

