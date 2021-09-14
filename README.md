# Transfer Learning Study on fMRI Images

<p align='center'>
<img src="https://github.com/kushaldev75/NMA-Deep-Learning/blob/main/plots/plot1.png" width=800 height=360></img>
</p>

This repository contains Short Research Project I did being part of [NMA Deep Learning Summer School, 2021](https://deeplearning.neuromatch.io/). Our project is titled as "Transfer Learning Study on fMRI Images". It is specifically focused on transfer learning in computer vision for medical imaging data.


# Abstract

<p>Deep learning has achieved significant results in multiple tasks, nonetheless the training procedure relies on large data and requires a lot of time. Transfer learning, which exchanges pretrained knowledge across different domains and tasks, has been demonstrated to be an effective method to solve these problems. In the medical domain, transfer learning still remains a great challenge due to several drawbacks such as the scarcity of data, the feature difference between natural and medical images, and ethical concerns. In this study, we investigate the effect of network architecture, particularly the classifier, into the adaptability of transfer learning. Our task is classifying the brain tumor into four categories with brain MRI images: no tumor, glioma tumor, meningioma tumor, and pituitary tumor. Our network is based on the VGG network because it provides multiple depth architecture and is simple enough to analyze the effect of each variation. From empirical results, we got promising results around 83% with VGG11 and batch normalization, demonstrating the usefulness of our study. Nevertheless, he found that more complex architectures such as VGG16 or VGG19 do not represent a considerable improvement to the transfer learning task. We suggest further studies in augmentation techniques, which could be used to overcome certain overfitting</p>

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

<p align='center'><br><br><b><I>“Whether we can optimally do knowledge transfer from natural images domain to a medical domain and maximize performance”</I></b><br><br></p>

## Dataset 

In order to address this question empirically. 

- We framed the problem as a Multiclass classification task. For this we used Brain Tumor fMRI classification dataset available on [kaggle](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri). 

- Unfortunately, there is discrepancy about the exact source of this dataset but on further exploration we found that most of the images are taken from the brain T1-weighted CE-MRI dataset which is open sourced on [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427). 

- The images have an in-plane resolution of 512 by 512. And this dataset is built by a collection of 3064 slices from 233 patients containing 3 tumor classes namely meningioma, glioma, pituitary tumor and one no tumor class

## Results

Observations:
1.   The architectures with batch normalization seems to behave worse than the conventional architectures at least on the initial iterations because they got lower accuracy in both train and test dataset.
2. 	However, the batch normalization architectures seem to reduce the overfitting gap between train and validation curves.

<p align='center'>
<img src="https://github.com/kushaldev75/NMA-Deep-Learning/blob/main/plots/plot1.png" width=600 height=300></img>
</p>

The same behavior with respect to the gap and accuracy is maintained in this experiments. However, now we got less accuracy than the expected, which suggests that the addition of additional linear layers does not contributes to the performance of the method.

<p align='center'>
<img src="https://github.com/kushaldev75/NMA-Deep-Learning/blob/main/plots/plot2.png" width=600 height=300></img>
</p>

We reported our best accuracy and loss with the VGG 11 network. The significant differences between prior trials are the number of layers to decrease features, batch size, and data augmentation. 

First, it was proposed that a few classifier layers were hard to learn new data features. Therefore, we decrease feature numbers smoothly with three linear layers and the batch sizes for better training. 

Furthermore, to prohibit overfitting problems, we augmented the dataset by random rotation for 30 degrees. Since the brain images are symmetric and have some direction information, we set the degree only to 30, which is widely acceptable for medical image data augmentation.

Best Model Hyperparameters:

- Model architecture: VGG11 feature extractor 25088 -> 2048 (Dropout-0.7) -> 256 (Dropout-0.6) -> 4
- Adam optimizer with learning rate: 1e-4
- batch size of 32
- number of epochs: 20
- Augmentation: RandomRotation(degrees=30) (from torchvision.transforms)


Best Model Performance:

- Accuracy: 84.26%
- Precision: 92.01%
- Recall: 84.26%
- F1-Score: 85.64%


<p align='center'>
<img src="https://github.com/kushaldev75/NMA-Deep-Learning/blob/main/plots/plot3.PNG" width=300 height=300></img>
</p>

## Conclusion and Further Study

- By acheiving the best model accuracy over 84%, our model in some way achieves the proposed goal and validates our hypothesis that we can successfully make transfer of knowledge to medical imaging domain. 
- However, we did not overcome the overfitting gap, which clearly can be appreciate as a challenging task in this problem.
- We suggest also to evaluate the model with more representative metrics such as Recall, Precision, and F1-score

From the results, networks with few layers show better accuracy than more layers. Also, we expect the batch normalization could improve the performance, but it didn’t. With this result, our conclusion is that the light model shows better results than complex models. We expect that the simple model is better as it is robust to data noise and flexible enough to learn new features. 


## Acknowledgement

It would not have been possible to do such a project in short time span of 18 days of Summer School without the support of my wonderful teammates Woohyun Eum and Oscar Guarnizo and teaching assistants Sean Bryne and Jospeh Donovan. Also I would like to thank NMA for giving opportunity and platform to gain this wonderful research oriented project experience. It was a great fun and learning experience!
