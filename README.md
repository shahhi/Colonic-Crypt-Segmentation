# Colonic-Crypt-Segmentation

Trained a segmentation model UNet using Pytorch to segment the colonic crypts in the tissue images. 

###### Dataset source: https://drive.google.com/drive/folders/1m-rYzhWbabhVBEMbClq6fRoOpLUgpFnx?usp=sharing

The colon dataset consists of 6 .tiff whole slide images (WSIs) and their GeoJSON annotations of colonic crypts. Each image is from a hematoxylin and eosin (H&E) stained coverslip from different regions of the colon (ascending, transverse, descending, and descending sigmoid). Hematoxylin and eosin stains nucleic acids deep blue-purple and nonspecific proteins varying degrees of pink, respectively. The WSIs were annotated by a pathologist using QuPath, and the resulting annotations were exported to GeoJSON format.

The position and shape of a crypt is represented by a set of pixel coordinates that indicate the outline of a single annotation. Run-length encodings (RLEs, see figure below for example) of the annotations are provided (train.csv and test.csv files), as well as metadata on the images and tissue donors (colon-dataset_information.csv).


## Data Preprocessing
<table>
<tr>
    <td><img src="Images/Sample_Image.png" width="520px"></td>
    <td><img src="Images/Sample_Mask.png" width="520px"></td>
    
</tr>
</table>



## Architecture
<img src="https://www.mdpi.com/sensors/sensors-22-00867/article_deploy/html/images/sensors-22-00867-g004.png"  width="520px">


## Training

## Inference

## Data Visualization

### PCA

<img src="Images/PCA.jpg" width="520px">
<table>
<tr>
    <td><img src="Images/TrainsetPCA.jpg" width="520px"></td>
    <td><img src="Images/TestsetPCA.jpg" width="520px"></td>
</tr>
</table>

### T-SNE
<img src="Images/TSNE.jpg" width="520px">
<table>
<tr>
    <td><img src="Images/TrainsetTSNE.jpg" width="520px"></td>
    <td><img src="Images/TestsetTSNE.jpg" width="520px"></td>
</tr>
</table>
    
### UMAP
    
<img src="Images/UMAP.jpg" width="520px">
<table>
<tr>
    <td><img src="Images/TrainsetUMAP.jpg" width="520px"></td>
    <td><img src="Images/TestsetUMAP.jpg" width="520px"></td>
</tr>
</table>

### MDS

<img src="Images/MDS.jpg" width="520px">
<table>
<tr>
    <td><img src="Images/TrainsetMDS.jpg" width="520px"></td>
    <td><img src="Images/TestsetMDS.jpg" width="520px"></td>
    
</tr>
</table>

## Insights

<img src="Images/Failedcase1.png" width="520px">

<img src="Images/Failedcase2.png" width="520px">
 
<img src="Images/Failedcase3.png" width="520px">

<img src="Images/Failedcase3.png" width="520px">
 
## Conclusion

<img src="Images/newplot.png" width="520px">

<img src="Images/newplot2.png" width="520px">

<img src="Images/newplot5.png" width="520px">

## References

[1] https://github.com/cns-iu/ccf-research-kaggle-2021

[2] 

## Paper 1: Summary
source: https://www.nature.com/articles/s41592-019-0403-1

## Paper 2: Summary
source: https://distill.pub/2019/activation-atlas/
