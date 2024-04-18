# Concept-QA+V-IP

**Aditya Chattopadhyay<sup>1</sup>, Kwan Ho Ryan Chan<sup>2</sup>, and René Vidal<sup>2</sup>** <br>

**¹Johns Hopkins University, USA, `achatto1 <at> jhu.edu`**

**²University of Pennsylvania, USA, `{ryanckh, vidalr} <at> seas.upenn.edu`**

Official code to accompany the paper [Bootstrapping Variational Information Pursuit with Large Language and Vision Models for Interpretable Image Classification (ICLR 2024)](https://openreview.net/forum?id=9bmTbVaA2A&noteId=IzZAQqhXHg).

## Overview
<p align="center">
<img src="./assets/teaser.png" alt="teaser.png" width="500"/>
</p>

Variational Information Pursuit (V-IP) is an interpretable-by-design framework that makes predictions by sequentially selecting a short chain of user-defined, interpretable queries about the data that are most informative for the task. The prediction is based solely on the obtained query answers, which also serve as a faithful explanation for the prediction. Applying the framework to any task requires (i) specification of a query set, and (ii) densely annotated data with query answers to train classifiers to answer queries at test time. This limits V-IP's application to small-scale tasks where manual data annotation is feasible. In this work, we focus on image classification tasks and propose to relieve this bottleneck by leveraging pretrained language and vision models. Specifically, following recent work, we propose to use GPT, a Large Language Model, to propose semantic concepts as queries for a given classification task. To answer these queries, we propose a light-weight Concept Question-Answering network (Concept-QA) which learns to answer binary queries about semantic concepts in images. We design pseudo-labels to train our Concept-QA model using GPT and CLIP (a Vision-Language Model). Empirically, we find our Concept-QA model to be competitive with state-of-the-art VQA models in terms of answering accuracy but with an order of magnitude fewer parameters. This allows for seamless integration of Concept-QA into the V-IP framework as a fast-answering mechanism. We name this method Concept-QA+V-IP. Finally, we show on several datasets that Concept-QA+V-IP produces shorter, interpretable query chains which are more accurate than V-IP trained with CLIP-based answering systems.


## Requirements
Please check out `requirements.txt` for detailed requirements. Overall, our code uses basic operations and do not require the latest version of PyTorch or CUDA to work. We also use `wandb` to moderate training and testing performance. One may remove lines related to `wandb` and switch to other packages if they desire. 
