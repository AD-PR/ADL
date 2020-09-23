![ADPR Logo](/figures/logo_ADPR.PNG)

# Alzheimer Disease Prognosis (AD-PR) 

**[Saturdays AI](https://www.saturdays.ai/) 3rd Edition**

A Deep Learning approach for Alzheimer Disease early diagnosis and prognosis, relying only on non-invasive procedures - structural MRIs and clinical data of the subjects.

This project aims to be a significant step-up in the Alzheimer prognosis field, a valuable support tool for medical professionals and a generalist research line for neurodegenerative diseases

Medium Article HERE

## Overview

This repository provides the experimentation code for Alzheimer Disease risk vs Healthy state binary prediction. All scripts can be found at the folder "Scripts", under the following structure:

* main.py is an example to allow running the full model.
* "Preprocessing" folder contains the code needed to:
  * Preprocess Structural 3D MRI images: normalization, scalation and fitting to standard Structural Brain Atlases.
  * Preprocess and order raw clinical data
* "Model" folder contains all the scripts related to the neural network architecture.

Project code can be easilly adapted for predicting other states encountered within the [ADNI repository](http://adni.loni.usc.edu/), as well as being expandable to other neurodegenerative diseases prediction purposes.

## Architecture

The structure of the deep learning neural network proposed here is based on the constraint of dealing with so few examples for training (up to 1000)

Therefore, a strategy of parametric efficiency is followed, with the objective of maintaining a deep network capable of learning the necessary features while preventing overfitting during learning. This is achieved with:

1. Residual connections to preserve variability
2. Custom Separable 3D Convolutional layers (far less parameters than normal 3DConv)
3. Few Shot Learnig strategy: Triplet Semi Hard Loss (clustering)

![ADPR Scheme](/figures/AD_PR_Scheme.png)

## Results

The following table shows the current metrics of this binary classification, revealing close to state-of-the-art accuracy.

METRIC | MRI + CLINICAL DATA | MRI 
------------ | ------------- | -------------
Train Accuracy | 0.867 | 0.867
Val Accuracy | 0.782 | 0.782
**Test Accuracy** | **0.794** | **0.794**
Sensitivity | 0.866 | 0.
Specificity | 0.736 | 0.
False Positive Rate | 0.263 | 0.
**False Negative Rate** | **0.133** | **0.**

Exploring other alternatives and with room for improvement, any contribution will be welcome.

## License

Authors:

**Cayetano Martínez-Muriel**: https://github.com/cayetanomarmur<br/>
**Carlos Morales Bartolomé**: https://github.com/CarlosMoralesB <br/>
**Carlos Sanmiguel Vila**: https://github.com/sanmi90


MRI 3D Images and clinical data obtained from [ADNI repository](http://adni.loni.usc.edu/)<br/>
Project developed within the framework of [Saturdays AI](https://www.saturdays.ai/) (Madrid, Spain, 3rd Edition): https://github.com/SaturdaysAI
