# Machine Learning Operations project
Eva Kaštelan - s232469 <br/>
Zeljko Antunovic - s233025 <br/>
Vilim Branica - s243169 <br/>
Nandor Takacs - s232458 <br/>
Beatriz Braga - s233576

## Project description
This group project is part of the Machine Learning Operations course at [DTU:02476](https://skaftenicki.github.io/dtu_mlops/projects/) for group 28 in January 2025. The project focuses on using machine learning to solve a face recognition problem by selecting, fine-tuning, evaluating and deploying a machine learning model. 

### Goal
The primary goal of this project is to develop and deploy a machine learning model that can:
- Recognize and identify individuals based on facial images.
- Perform efficiently with a relatively small and balanced dataset.

### Framework
We used [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html) to develop our model. Furthermore, we are going to utilize [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training, validating and testing our model.

### Data
Our dataset is sourced from a publicly available repository: Thinking Neuron. The dataset contains:
- Approximately 300 images representing 16 unique individuals.
- Approximately the same number of images per person, ensuring balance and fairness during training.

The dataset is suitable for transfer learning due to its small size, allowing us to leverage the power of pre-trained models for feature extraction and classification.

### Model
We chose to fine-tune a pre-trained ResNet34 model from the [torchvision library](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html). The ResNet34 model was originally trained on the ImageNet1k dataset, which consists of 1.2 million labeled images across 1,000 classes. This pretraining provides a strong foundation for transfer learning, especially for tasks with limited data like ours.

Why ResNet34?
- Residual Connections: ResNet34 utilizes residual blocks, which help mitigate the vanishing gradient problem and enable deeper networks to learn effectively.
- Pretraining Benefits: The model’s pretrained weights on ImageNet1k allow it to extract generic image features that are transferable to our face recognition task.
- Efficiency: ResNet34 strikes a balance between performance and computational efficiency, making it suitable for deployment on limited hardware resources.
