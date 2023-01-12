# Clean data files used in NER (Named Entity Recognition)

## Table of Contents

[comment]: <> (- [Security]&#40;#security&#41;)
* [Functions](#Functions)
* [Usage](#Usage)
* [Notes](#Notes)


[comment]: <> (- [Usage]&#40;#usage&#41;)
[comment]: <> (- [API]&#40;#api&#41;)
[comment]: <> (## Security)

## Functions
Currently, we only provides pre-processing algorithms including:
1. Remove html websites, web marks 
2. Split very long entities 
3. Split long text input 

## Usage

1. Put your own data files in Data folder 
2. Specify parameters like max_len... 
3. Execute "DataProcessor.py"
4. You can add your own operators into this project


## Notes
1. This script is used only in NER task, to preprocess the train/dev/test data.
2. The default format of the data files is json format, here we use 
CBLUE competition files as examples. Other formats need to be converted firstly.