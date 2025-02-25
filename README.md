# Prithvi Change Detection

An experiment to test the use of the pre-trained [Prithvi model](https://huggingface.co/ibm-nasa-geospatial) for a change detection application

The test case is the [xBD dataset](https://arxiv.org/abs/1911.09296) containing before and after imagery of building damage caused by natural disasters.

The idea is to use the the pre-trainied Prithvi encoder to embed the "pre" and "post" image sequence and train a semantic change detection head that takes the embedding as input.

We can compare performance to the [AB2CD](https://arxiv.org/abs/2309.01066) results on the same data.


# Setup

## Install

    pip install -r requirements.txt

## Model

Get the model code:

    git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M
    mv Prithvi-100M prithvi_100m

Download the checkpoint file, `Prithvi_100M.pt`, manually from: https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/tree/main 

Make it importable:

    touch prithvi_100m/__init__.py

## Data

Download the data set and follow the instructions at: https://xview2.org/download


## Create masks

    python create_dataset.py target-masks path/to/data

# NOTES

Sentinal bands in HLS data used to train Prithvi:

```
B02: blue, 490 nm
B03: green, 560 nm
B04: red, 665 nm
B8A: IR, 865 nm
B11: SWIR, 1610 nm
B12: SWIR, 2190 nm
```

**Loss Scaling**
1. Over the dataset, compute the pixel area of each class for each image. (Done in create_dataset.metadata and stored in metadata.csv)
2. Sum the pixel area of each class over the whole dataset.
3. Find total pixel area of the dataset, `total_area = num_images * 1024*1024`
4. Calculate the loss weights as:
```
inverse_probability = 1 / (class_areas / total_area)
weights =  inverse_probability / (inverse_probability.sum())
```