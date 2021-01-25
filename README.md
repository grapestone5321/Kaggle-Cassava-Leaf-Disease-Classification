# Kaggle-Cassava-Leaf-Disease-Classification
Kaggle-Cassava-Leaf-Disease-Classification

## End Date (Final Submission Deadline): 
### February 18, 2021 11:59 PM UTC


-------

## Task
### Your task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. 
 
With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

-------

## Four of the most common cassava diseases 


      "root":{5 items
      "0":string"Cassava Bacterial Blight (CBB)"
      "1":string"Cassava Brown Streak Disease (CBSD)"
      "2":string"Cassava Green Mottle (CGM)"
      "3":string"Cassava Mosaic Disease (CMD)"
      "4":string"Healthy"
      }


## Papers:

## Cassava Disease
### Managing cassava virus diseases in Africa
http://www.fao.org/fileadmin/user_upload/emergencies/docs/RCI%20Cassava%20brochure_ENG_FINAL.pdf


## Cassava Brown Streak Disease (CBSD): 

### Cassava brown streak disease: a threat to food security in Africa
http://biblio1.iita.org/bitstream/handle/20.500.12478/1151/U15ArtPatilCassavaInthomDev.pdf?sequence=1&isAllowed=y

## Cassava Mosaic Disease (CMD): 

### Resistance to cassava mosaic disease in transgenic cassava expressing antisense RNAs targeting virus replication genes
https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-7652.2005.00132.x

## Cassava Bacterial Blight (CBB): 

### Genetic mapping of resistance to bacterial blight disease in cassava (Manríhof esculenfa Crantz)
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.469.2396&rep=rep1&type=pdf

## Cassava Green Mite (CGM):

### Genome-Wide Association Study of Resistance to Cassava Green Mite Pest and Related Traits in Cassava
https://acsess.onlinelibrary.wiley.com/doi/pdf/10.2135/cropsci2018.01.0024




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

## Cassava Disease Classification: InClass Prediction Competition
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



## iCassava 2019
https://sites.google.com/view/fgvc6/competitions/icassava-2019

### github: FGVCx Cassava disease diagnosis
https://github.com/icassava/fgvcx-icassava



-------


## Pytorch
https://pytorch.org/

### SAVING AND LOADING MODELS
https://pytorch.org/tutorials/beginner/saving_loading_models.html

### ResNet in PyTorch - GitHub
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

### EfficientNet: Improving Accuracy and Efficiency through AutoML and Model Scaling
https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html

## Paper:

### Deep Learning with PyTorch
https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf


-------

## Paper-1

### Benchmark Analysis of Representative Deep Neural Network Architectures
https://arxiv.org/pdf/1810.00736.pdf


### MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://arxiv.org/pdf/1704.04861.pdf


-------

## Paper-2: Augmentation, Test-time augmentation (or TTA)


### Albumentations: fast and flexible image augmentations
https://arxiv.org/pdf/1809.06839.pdf



### The Effectiveness of Data Augmentation in Image Classification using Deep Learning
https://arxiv.org/pdf/1712.04621.pdf


### When and Why Test-Time Augmentation Works
https://dmshanmugam.github.io/pdfs/tta_2020.pdf

### When and Why Test-Time Augmentation Works
https://arxiv.org/pdf/2011.11156.pdf


### Test-time augmentation with uncertainty estimation for deep learning-based medical image segmentation
https://openreview.net/pdf?id=Byxv9aioz


### Learning Loss for Test-Time Augmentation
https://arxiv.org/pdf/2010.11422.pdf


### Greedy Policy Search: A Simple Baseline for Learnable Test-Time Augmentation
https://arxiv.org/pdf/2002.09103.pdf

--------

## Blogs

## Test-time augmentation, or TTA for short

### How to Use Test-Time Augmentation to Make Better Predictions:
https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/


### Test-Time Augmentation For Tabular Data With Scikit-Learn:
https://machinelearningmastery.com/test-time-augmentation-with-scikit-learn/

-------

## Cross-Validation Strategy

## Blogs

### Complete guide to Python’s cross-validation with examples
https://towardsdatascience.com/complete-guide-to-pythons-cross-validation-with-examples-a9676b5cac12

- github: vaasha/Machine-leaning-in-examples

https://github.com/vaasha/Machine-leaning-in-examples/blob/master/sklearn/cross-validation/Cross%20Validation.ipynb

### sklearn.model_selection.train_test_split
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


## Papers:

### Permutation Tests for Studying Classifier Performance
https://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf


-------

## Things may help public lb score: Discussion by Heroseo
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
      RAdam+Lookahead is also good. Other than that, testing was required, 
      but there was no significant change in public lb score.

- Relabeling and Finetuning

      CV improved a lot, but public lb did not change much.
      (The CV was very bad when only training the dataset that my model thought was noise.)

- Criterion (loss function)

      Label smoothing is good in public lb. 
      Bi-Tempered Logistic Loss and Focal Cosine Loss can be a good alternative.

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

## Light augs to 0.906: Discussion by zlannn

https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/206724

      Yesterday, I shared the opinion here https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/206489 
      that we might need carefully select TTA augs or even augs while training.

      I rarely train a model with almost no augs in CV task because regular augs always help.

      However, I noticed light augs in this task leads to an obvious boost on LB (did not boost that much on CV).

      Therefore, I trained a light aug version model today, and it boosts my LB to the 1st.

      Never been such a high rank before. 
      Although the CV score indicates that it is likely to be overfitting on public LB. 
      I think it still worth sharing for those who wanna climb a higher place.

-------

## Progress
### Current Best LB Score: 0.902

-------


### Pytorch Efficientnet Baseline [Inference] TTA
https://www.kaggle.com/mekhdigakhramanian/pytorch-efficientnet-baseline-inference-tta


-------

## Process: Pytorch Efficientnet Baseline [Inference] TTA

      package_path = '../input/pytorch-image-models/pytorch-image-models-master' 
             
      train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
      
      submission = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')


## Helper Functions

      img = get_img('../input/cassava-leaf-disease-classification/train_images/1000015157.jpg')

### def seed_everything(seed):

      def seed_everything(seed):
          random.seed(seed)
          os.environ['PYTHONHASHSEED'] = str(seed)
          np.random.seed(seed)
          torch.manual_seed(seed)
          torch.cuda.manual_seed(seed)
          torch.backends.cudnn.deterministic = True
          torch.backends.cudnn.benchmark = True
 
### def get_img(path):
 
      def get_img(path):
         im_bgr = cv2.imread(path)
         im_rgb = im_bgr[:, :, ::-1]
         return im_rgb         
          
    
## Dataset

      class CassavaDataset(Dataset):
          def __init__(
              self, df, data_root, transforms=None, output_label=True
          ):
        
              super().__init__()
              self.df = df.reset_index(drop=True).copy()
              self.transforms = transforms
              self.data_root = data_root
              self.output_label = output_label
    
          def __len__(self):
              return self.df.shape[0]
    
          def __getitem__(self, index: int):
        
              # get labels
              if self.output_label:
                  target = self.df.iloc[index]['label']
          
              path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        
              img  = get_img(path)
        
              if self.transforms:
                  img = self.transforms(image=img)['image']
            
              # do label smoothing
              if self.output_label == True:
                  return img, target
              else:
                  return img



## Define Train\Validation Image Augmentations

### def get_train_transforms():

      def get_train_transforms():
          return Compose([
                  RandomResizedCrop(CFG['img_size'], CFG['img_size']),
                  Transpose(p=0.5),
                  HorizontalFlip(p=0.5),
                  VerticalFlip(p=0.5),
                  ShiftScaleRotate(p=0.5),
                  HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                  RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                  CoarseDropout(p=0.5),
                  Cutout(p=0.5),
                  ToTensorV2(p=1.0),
              ], p=1.)


### def get_valid_transforms():

      def get_valid_transforms():
          return Compose([
                  CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
                  Resize(CFG['img_size'], CFG['img_size']),
                  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                  ToTensorV2(p=1.0),
              ], p=1.)

### def get_inference_transforms():

      def get_inference_transforms():
          return Compose([
                  RandomResizedCrop(CFG['img_size'], CFG['img_size']),
                  Transpose(p=0.5),
                  HorizontalFlip(p=0.5),
                  VerticalFlip(p=0.5),
                  HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, 
                                           val_shift_limit=0.2, p=0.5),
                  RandomBrightnessContrast(brightness_limit=(-0.1,0.1), 
                                           contrast_limit=(-0.1, 0.1), p=0.5),
                  Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225], 
                                           max_pixel_value=255.0, p=1.0),
                  ToTensorV2(p=1.0),
              ], p=1.)

## Model

### Main Loop

        valid_ = train.loc[val_idx,:].reset_index(drop=True)
        valid_ds = CassavaDataset(valid_, 
                                  '../input/cassava-leaf-disease-classification/train_images/', 
                                  transforms=get_inference_transforms(), output_label=False)
        
        
        test = pd.DataFrame()
        test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
        test_ds = CassavaDataset(test, 
                                 '../input/cassava-leaf-disease-classification/test_images/', 
                                 transforms=get_inference_transforms(), output_label=False)



        for i, epoch in enumerate(CFG['used_epochs']):    
            model.load_state_dict(torch.load('../input/fork-pytorch-efficientnet-baseline-train-amp-a/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)))
            
            
-------

### CFG = {

### tta: default= 3    
    
    'tta': 1   LB 0.896    ver14  
    'tta': 2   LB 0.902    ver13  --- Best
    'tta': 3   LB 0.898    ver6 
    'tta': 4   LB 0.898    ver15  
    
### lr: default= 1e-4

tta = 3:

       'lr': 1e-3   LB 0.898    ver8       
       'lr': 1e-4   LB 0.898    ver6     
       'lr': 1e-5   LB 0.898    ver7     


### epochs: default= 32


tta = 3:

    'epochs': 16     LB 0.898    ver9     
    'epochs': 32     LB 0.898    ver6      

### img_size: default= 512

tta = 3:

    'img_size':  128    LB 0.602    ver10  
    'img_size':  256    LB 0.878    ver12 
    'img_size':  512    LB 0.898    ver6
    'img_size': 1024    LB error    ver11

tta = 2:

    'img_size': 384    LB 0.896    ver36
    'img_size': 448    LB 0.898    ver38
    'img_size': 480    LB 0.898    ver39
    'img_size': 496    LB          ver
    'img_size': 512    LB 0.902    ver13  --- Best    
    'img_size': 576    LB 0.901    ver21


### fold_num: default= 10

 tta = 3:
 
    'fold_num':  5      LB 0.898    ver18
    'fold_num': 10      LB 0.898    ver6

tta = 2:

    'fold_num':  2      LB 0.902    ver20
    'fold_num': 10      LB 0.902    ver13  --- Best
    'fold_num': 15      LB 0.902    ver19
 
 
### valid_bs:  default= 32

tta = 2:

      'valid_bs':  16     LB 0.899    ver25
      'valid_bs':  28     LB 0.896    ver26
      'valid_bs':  32     LB 0.902    ver13,22  --- Best
      'valid_bs':  36     LB 0.900    ver27
      'valid_bs':  64     LB 0.897    ver23
   

### seed:  default= 719

tta = 2:

    'seed': 　42     LB 0.895    ver28
    'seed':  500     LB 0.895    ver31
    'seed':  600     LB 0.898    ver33
    'seed':  700     LB 0.899    ver34    
    'seed':　719     LB 0.902    ver13,22  --- Best
    'seed':  800     LB 0.897    ver32 
    'seed': 1000     LB 0.897    ver30
    'seed': 1500     LB 0.897    ver29


### model_arch:  default= tf_efficientnet_b3_ns

tta = 2:, 

    'model_arch': 'tf_efficientnet_b3_ns'
               'img_size': 480      LB 0.898    ver39
               'img_size': 512      LB 0.902    ver13,22  --- Best
    
    'model_arch': 'tf_efficientnet_b4_ns'
               'img_size': 480      LB 0.895    ver40       
               'img_size': 512      LB 0.896    ver41      
 
### used_epochs: default= [6,7,8,9],

tta = 2: 

       'used_epochs': [5,6,7,8]        LB 0.899    ver43
       'used_epochs': [6,7,8,9]        LB 0.902    ver13,22,44  --- Best
       'used_epochs': [7,8,9,10]       LB 0.897    ver42
       'used_epochs': [8,9,10,11]      LB 0.896    ver45
       
### weights:  default= [1,1,1,1]   

tta = 2: 
      
       'weights': [1,1,1,0]       LB 0.899    ver46
       'weights': [1,1,1,0.5]     LB 0.901    ver48
       'weights': [1,1,1,0.7]     LB 0.901    ver49
       'weights': [1,1,1,1]       LB 0.902    ver13,22,44  --- Best  
       'weights': [1,1,1,1.2]     LB 0.902    ver50
       'weights': [1,1,1,2]       LB 0.900    ver47
       'weights': [1,1,1,2,1]     LB 0.900    ver53
       'weights': [1,1,1.2,1]     LB 0.900    ver54

### fold > n:  default= n

      folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)    
          for fold, (trn_idx, val_idx) in enumerate(folds):
              # we'll train fold 0 first       
              if fold > 0:
                  break 
              
              if fold > 0:         LB 0.902    ver13,22,44  --- Best  
              if fold > 1:         LB 0.896    ver56
              if fold > 10:        LB 0.898    ver57
              
    
     
 -------  
    
## Cassava classification 6
https://www.kaggle.com/salmaneunus/cassava-classification-6

### CFG = {

### tta: default= 6    
    

    'tta': 2   LB 0.900    ver2  
    'tta': 6   LB 0.901    ver1 

### lr: default= 1e-2

    'lr': 1e-2   LB 0.901    ver1 
    'lr': 1e-3   LB 0.901    ver4  


-------
