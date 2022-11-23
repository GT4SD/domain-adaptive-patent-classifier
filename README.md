# Patent classification

Patent classification using various transformer and/or CNN based models.

### Setup
```buildoutcfg
conda env create -f conda.yml
conda activate dapc
```

### Train/Evaluate/Predict

Check the provided [notebooks](./notebooks) of how to use the module for training, evaluation and prediction. Alternatively, you can use the existing [scripts](./scripts) to perform k-fold Cross-validation of the models.


### Data

The English and multilingual versions of the Crop Protection Industry dataset can be accessed on this [Box link](https://ibm.box.com/v/dapc-dataset). You can reproduce the results of our paper using these datasets and the provided scripts. If you use the data, please cite our work. 


### Models

In the context of this work, we trained three BERT-like models based on the domain adaptive pretraining method for the patent domain. These models used as base models for our patent classification work but they can be also used for any other patent related NLP task. We have uploaded these models to the HuggingFace model hub and can be accessed on the following links: [dapBERT](https://huggingface.co/christofid/dapbert), [dapSciBERT](https://huggingface.co/christofid/dapscibert) and [dapBERT-multi](https://huggingface.co/christofid/dabert-multi). Further information regarding the models can be found in our paper. If you use these models, please cite our work. 
