# Colonic-Crypt-Segmentation

Trained a segmentation model UNet using Pytorch to segment the colonic crypts in the tissue images. 

#### Dataset source: https://drive.google.com/drive/folders/1m-rYzhWbabhVBEMbClq6fRoOpLUgpFnx?usp=sharing

The colon dataset consists of 6 .tiff whole slide images (WSIs) and their GeoJSON annotations of colonic crypts. Each image is from a hematoxylin and eosin (H&E) stained coverslip from different regions of the colon (ascending, transverse, descending, and descending sigmoid). Hematoxylin and eosin stains nucleic acids deep blue-purple and nonspecific proteins varying degrees of pink, respectively. The WSIs were annotated by a pathologist using QuPath, and the resulting annotations were exported to GeoJSON format.

The position and shape of a crypt is represented by a set of pixel coordinates that indicate the outline of a single annotation. Run-length encodings (RLEs, see figure below for example) of the annotations are provided (train.csv and test.csv files), as well as metadata on the images and tissue donors (colon-dataset_information.csv).


## Data Preprocessing

<table>
<tr>
    <td><img src="Images/Sample_Image.png" width="520px"></td>
    <td><img src="Images/Sample_Mask.png" width="520px"></td>
    
</tr>
</table>


Due to large size (4536 x 4704 x 3) of image to save computational power decided to generate patch sizes of 512 x 512. Saved the path of each patch, path to its mask and annoted class (1 crypt 0 background) in a train_data.csv file for easy access.


```bash
# Used to extract patches of size = (512, 512) with no overlapping starting from left top as well as right bottom
def extract_patches
    """
    This Method creates a .csv to containing image and mask paths
    
    args:
    df_train = dataframe containing image Ids and RLE annotations
    image_data_dir  = Path to the train image folder
    mask_data_dir  = Path to the train mask folder
    train_patches_directory = Path to the folder where we want to save the patches
    patch_size = patch size
    
    Returns: None
    """
```


## Architecture

<p align="center">
    MODEL USED: UNET with EFFICIENTNET-B2 as ENCODER trained on IMAGENET dataset
<img src="https://www.mdpi.com/sensors/sensors-22-00867/article_deploy/html/images/sensors-22-00867-g004.png"  width="520px">
</p>

## Training

```bash
# imported to use UNET architecture with encoder pretrained 
import segmentation_models_pytorch as sm
# imported to use Ranger(Adam + LookAhead) as an optimizer
import torch_optimizer as t_optim
# imported for data augmentation
import albumentations as A
```

```bash
BATCH_SIZE = 32
INPUT_CHANNELS = 3
INPUT_SHAPE = (512,512,3)
PATIENCE = 5 # Implemented early stopping
EPOCHS = 50
KFOLD = 2 # KFold cross validation
LOSS = Cross Entropy Loss
DICE LOSS
DICE SCORE
```
```bash
train_transform =  A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.OneOf([
                A.ElasticTransform(p=.3),
                A.GaussianBlur(p=.3),
                A.GaussNoise(p=.3),
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(15,25,0),
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            ], p=0.3),

        ])
```

## Inference

*

```bash
# Dice score for test set
            Image                          Score
1 CL_HandE_1234_B004_bottomleft     0.8751588501232356
2 HandE_B005_CL_b_RGB_bottomleft    0.7151214544866026

Average	0.7951401523049191
```
```bash
# Dice score for train set
            Image                          Score
1 CL_HandE_1234_B004_bottomright    0.7902297684303141
2 CL_HandE_1234_B004_topleft	    0.8143473842064496
3 CL_HandE_1234_B004_topright	    0.8089536289263017
4 HandE_B005_CL_b_RGB_bottomright   0.8302962795308487
5 HandE_B005_CL_b_RGB_topleft	    0.8790617750363643

Average	0.8245777672260557
```
## Data Visualization

### PCA

<p align="center">
<img src="Images/PCA.jpg" width="520px">
</p>
<table>
<tr>
    <td><img src="Images/TrainsetPCA.jpg" width="520px"></td>
    <td><img src="Images/TestsetPCA.jpg" width="520px"></td>
</tr>
</table>

### T-SNE

<p align="center">
<img src="Images/TSNE.jpg" width="520px">
</p>
<table>
<tr>
    <td><img src="Images/TrainsetTSNE.jpg" width="520px"></td>
    <td><img src="Images/TestsetTSNE.jpg" width="520px"></td>
</tr>
</table>
    
### UMAP

<p align="center">
<img src="Images/UMAP.jpg" width="520px">
</p>
<table>
<tr>
    <td><img src="Images/TrainsetUMAP.jpg" width="520px"></td>
    <td><img src="Images/TestsetUMAP.jpg" width="520px"></td>
</tr>
</table>

### MDS

<p align="center">
<img src="Images/MDS.jpg" width="520px">
</p>
<table>
<tr>
    <td><img src="Images/TrainsetMDS.jpg" width="520px"></td>
    <td><img src="Images/TestsetMDS.jpg" width="520px"></td>
    
</tr>
</table>

## Insights

<p align="center">
<img src="Images/Failedcase1.png" width="520px">

<img src="Images/Failedcase2.png" width="520px">
 
<img src="Images/Failedcase3.png" width="520px">

<img src="Images/Failedcase3.png" width="520px">
    
</p>
 
## Conclusion

<p align="center">
<img src="Images/newplot.png" width="520px">

<img src="Images/newplot2.png" width="520px">

<img src="Images/newplot5.png" width="520px">
    
</p>

## References

[1] https://github.com/cns-iu/ccf-research-kaggle-2021

[2] https://lessw.medium.com/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

[3] https://github.com/j-sripad/Roof_segmentation (Few parts are adapted from my Computer Vision Final Project)

## Paper 1: Summary
source: https://www.nature.com/articles/s41592-019-0403-1


