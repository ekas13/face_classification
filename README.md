# Machine Learning Operations project

Eva Kaštelan s232469 <br/>
Zeljko Antunovic s233025 <br/>
Vilim Branica s243169 <br/>
Nandor Takacs s232458 <br/>

## Project description
This is a group project for the DTU course Machine Learning Operations [DTU:02476](https://skaftenicki.github.io/dtu_mlops/projects/).

### Goal
Our project revolves around creating and deploying a Machine Learning model for a face recognition problem.
The goal for our model is to recognize people based on photos of their faces.

### Framework
We used [PyTorch](https://pytorch.org/) to develop our model. Furthermore, we used a pretrained Resnet34 model from the torchvision library https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html. The Transformer framework performs approximately as well as some state-of-the-art Convolutional Neural Network models (CNNs), but requires much less time and computational resources to train.

### Data
Our dataset for the model is the Kaggle dataset [CelebFaces Attributes](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data). It consists of over 200k images of celebrities with 40 binary attribute annotations.

### Model
The model we chose to fine-tune with our dataset is [Vision Transformer(ViT)](https://huggingface.co/docs/transformers/model_doc/vit), which was pretrained on 14 million images and can classify 1000 different items. We could also have used other ViT-type models, such as [ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae) or [ViTMSN](https://huggingface.co/docs/transformers/model_doc/vit_msn). However, for this project we chose the elementary model.
