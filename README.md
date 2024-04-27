# Prithvi Change Detection

An experiment to test the use of the pre-trained [Prithvi model](https://huggingface.co/ibm-nasa-geospatial) for a change detection application

The test case is the [xBD dataset](https://arxiv.org/abs/1911.09296) containing before and after imagery of building damage caused by natural disasters.

The idea is to use the the pre-trainied Prithvi encoder to embed the "pre" and "post" image sequence and train a semantic change detection head that takes the embedding as input.

We can compare performance to the [AB2CD](https://arxiv.org/abs/2309.01066) results on the same data.

# TODO

1. Check on downsampling and zero filling missing bands
2. Normalization statistics?
3. Generate target masks from the "post" geojson labels
4. Create Dataset and Dataloaders

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


