# Kaggle-Cassava-Leaf-Disease-Classification
Kaggle-Cassava-Leaf-Disease-Classification

## End Date (Final Submission Deadline): 
### February 18, 2021 11:59 PM UTC


-------

## Task
Your task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. 
 
With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

-------

## Evaluation
### Submissions will be evaluated based on their categorization accuracy.

categorization accuracy: https://developers.google.com/machine-learning/crash-course/classification/accuracy

### Submission Format
The submission format for the competition is a csv file with the following format:

     image_id,label
     1000471002.jpg,4
     1000840542.jpg,4
     etc.

-------

## Four of the most common cassava diseases 
- Cassava Brown Streak Disease (CBSD) 

- Cassava Mosaic Disease (CMD) 

- Cassava Bacterial Blight (CBB) 

- Cassava Green Mite (CGM)

-------

## Cassava Disease Classification:
https://www.kaggle.com/c/cassava-disease/overview

Classify pictures of cassava leaves into 1 of 4 disease categories (or healthy)

InClass Prediction Competition

Fine-Grained Visual Categorization 6 87 teams 2 years ago

### 1st place solution:
https://www.kaggle.com/c/cassava-disease/discussion/94114

### 2nd place Solution:
https://www.kaggle.com/c/cassava-disease/discussion/94112

### 3rd place solution:
https://www.kaggle.com/c/cassava-disease/discussion/94102

-------


## pytorch
https://pytorch.org/

### SAVING AND LOADING MODELS
https://pytorch.org/tutorials/beginner/saving_loading_models.html

### ResNet in PyTorch - GitHub
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

### EfficientNet: Improving Accuracy and Efficiency through AutoML and Model Scaling
https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html


-------

## Paper

### Benchmark Analysis of Representative Deep Neural Network Architectures
https://arxiv.org/pdf/1810.00736.pdf



--------

## Things may help public lb score: Discussion
https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/207450

- Image Size

      Larger image size helps public lb score. 384 x 384 ~ 512 x 512 are enough.

- Augmentation

      Augmentation with flips and rotations are good for starter. Things like cutmix helps with CV score, but there is no significant change in public lb for me.

- Normalization

      Using mean, std for this competition helps a little boost in CV and public lb, instead of mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] for ImageNet. 
      But the mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] is enough for public lb scores.

- Model size

      A small model like resnext50 or efficientnet-b3 is doable for medals. However, keep the possibilities open for larger models.

- Cross-Validation Strategy

      The CV strategy is important. 
      5 folds is enough. 10 folds had an advantage in cv, but had no significant impact in public lb. 
      For each specific fold, the difference in scores for both cv and lb is very large. 
      Because of this, changing the seed may can help. Not using all folds helps public lb.
      (CV and public lb seem to be related to some extent.)

- Optimizer
      Adam is a good starting point. 
      RAdam+Lookahead is also good. Other than that, testing was required, but there was no significant change in public lb score.

- Relabeling and Finetuning

      CV improved a lot, but public lb did not change much.
      (The CV was very bad when only training the dataset that my model thought was noise.)

- Criterion (loss function)

      Label smoothing is good in public lb. Bi-Tempered Logistic Loss and Focal Cosine Loss can be a good alternative.

- TTA

      Choosing commonly used TTA, public lb score may get worse. 
      This is heavily influenced by the CV Strategy. 
      A little rotation or simple augmentation can help. 
      However, it is sensitive to the number of TTAs.
      (In fact, you can reach the current gold medal area with no TTA.)

- Ensemble

      It helps with public lb. 
      However, single model can get public lb score for current gold medal area.


-------

## Progress
### Current Best LB Score: 0.898
-------


### Pytorch Efficientnet Baseline [Inference] TTA

       'lr': 1e-3 LB 0.898 ver8

