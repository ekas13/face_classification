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

## Project structure

The directory structure of the project looks like this:
```txt
./
├── .dvc/
├── .git/
├── .github/
├── cloudbuild/
├── configs/
├── data/
├── dockerfiles/
├── models/
├── reports/
├── src/
├── tests/
├── .coveragerc
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── environment.yaml
├── pyproject.toml
├── requirements.txt
├── requirements_frontend.txt
├── requirements_tests.txt
└── tasks.py
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Where to access the deployed model?
The model is deployed and hosted on this URL:
https://app-docker-image-294894715547.europe-west1.run.app

The frontned is deployed and hosted on this URL:
https://frontend-294894715547.europe-west1.run.app/

Use the frontend to interact with the API.

## How to run the model locally
This was used internally for developing our project. The user must be connected to the GCP group and the Weights & Biases group.
### Setup the environment
First we want to create a new conda environment:
```
invoke create-environment
```
Then, install necessary packages:
```
invoke requirements
```
Finally, login into the WANDB account:
```
wandb login
```
### Fetch raw data from Google Cloud
Follow this for installation guide of gcloud CLI: https://cloud.google.com/sdk/docs/install (ask Eva or Zeljko if you need help).
Then install  ```dvc``` with ```pip install dvc```.
After you have ```dvc``` set up, run this command:
```
gcloud auth application-default login
```
Which should open your browser where you have to login with your google account that you use for Google Cloud Coupon. If you get an error when opening the link, run the command with ```--no-launch-browser``` flag and copy paste the link from the terminal instead.
Now, you can finally fetch the raw data by running:
```
dvc pull --no-run-cache
```
Your ```data/raw``` folder should now have ```train``` and ```test``` split.

### Preprocess the data
To preprocess the data, run:
```
invoke preprocess-data
```
which will create the ```processed``` folder with 3 subfolders in it, ```train```, ```test```,  ```val```. All images in these folders are resized to ```(256, 256)``` so they have a uniform size across the dataset.

### Train from scratch (Optional)
If you want to train the model from scratch locally, run this command:
```
invoke train
```

The summary of the training run will be given to you in the console with a link to the Weights & Biases report.
### Evaluating the model
To evaluate the model on the test set, run this command:
```
invoke evaluate
```

If you already have the model locally, then run this version:
```
invoke evaluate models/model.pth
```

### Running the API server & the frontend
Have two terminals open for this part. In the first terminal run the server with this command:
```
invoke server
```
In the second terminal, start your frontend with this command:
```
invoke frontend
```
Go to the frontend local URL and try out the API by uploading an image to it. Since the model was trained on 16 different people and it uses their faces to recognize them, for a relevant output upload an image from your data/processed/test folder.
