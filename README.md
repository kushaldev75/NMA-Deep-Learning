# Transfer Learning Study on fMRI Images

<p align='center'>
<img src="https://user-images.githubusercontent.com/74819807/133071295-b53d90b5-9b82-40c6-8052-334a20b109b5.png" width=800 height=360></img>
</p>

This repository contains Short Research Project I did being part of [NMA Deep Learning Summer School, 2021](https://deeplearning.neuromatch.io/). Our project is titled as "Transfer Learning Study on fMRI Images".

## Transfer Learning

In this study, we worked with Transfer learning. It allow us to exchange pre-trained knowledge across different domains and tasks, where the fundamental idea is to reuse pre-trained weights of a network and transfer to one. This method has been demonstrated to be really effective to solve problems related with scarcity of data, which is a great problem with fMRI images.

## Motivation

In the medical domain, transfer learning still remains a great challenge due to several drawbacks. 
Our motivation arises with the principal problem that is there are feature differences between natural images (of Image Net) and fMRI images
This problem conveys other one that intended to tackle.
1. First, we need to provide a large data to train an useful model from scratch, but commonly the medical data is scarce.
2. Many medical images could have high resolution, which provokes a high consumption of time and computational sources.
3. Finally, many of the well-known architectures suffers of overparameterization on ImageNet dataset, which hampers the transfer of learning to other domains.

## Research Question

During early stages as part of literature review we went through previous work done mainly on the topic of Transfer learning and Deep Learning in the Medical Imaging domain.
Interestingly, we discovered this NeurIPS 2019 paper titled “Transfusion: Understanding Transfer Learning for Medical Imaging'' by [Maithra Raghu et al, Google Brain](https://arxiv.org/abs/1902.07208). 

In this work, they also investigate some central questions for transfer learning in medical imaging tasks. And surprisingly, one of the conclusions they draw is that transfer learning does not significantly affect performance on medical imaging tasks, with models trained from scratch performing nearly as well as standard ImageNet transferred models.
Looking at this strong point made by them, we lead framing our Research Question as:

<p align='center'><b><I>“Whether we can optimally do knowledge transfer from natural images domain to a medical domain and maximize performance”</I></b></p>

## Dataset 

In order to address this question empirically. 

- We framed the problem as a Multiclass classification task. For this we used Brain Tumor fMRI classification dataset available on [kaggle](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri). 

- Unfortunately, there is discrepancy about the exact source of this dataset but on further exploration we found that most of the images are taken from the brain T1-weighted CE-MRI dataset which is open sourced on [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427). 

- The images have an in-plane resolution of 512 by 512. And this dataset is built by a collection of 3064 slices from 233 patients containing 3 tumor classes namely meningioma, glioma, pituitary tumor and one no tumor class

## Results

## Acknowledgement

It would not have been possible to do such project in short time span of 18 days without the support of my woderful teammates Woohyun Eum and Oscar Guarnizo and teaching assistants Sean Bryne and Jospeh Donovan. Also I would like to thank NMA for giving this wonderful research oriented project experience. It was a great fun and learning experience!
