# Retinal_Vessel_Segmentation
The vessels of retina in very important mark for curing various cardiovascular and opthalmological diseases , that's why automatic segmentation of vessel is necessary to cure and monitor over these chronic diseases and problem related to these diseases. Few year earlier ,  retinal vessel segmentation done by the method of deep learning have reached state of art performance. As we know there's lies lots of variation in retinal vessels which includes noisy backgorund , optic disk and critical issues related to thin vessels etc. So to overcome these problem , we came with an idea of a U-Net like model with weighted Res-Net. Our model is applied on publicly available DRIVE dataset and our proposal also uses data augmentation technique which allow us to explore the information of fundus image during training.

We have used publicly available DRIVE dataset in which we have training and test dataset . Training consists of 20 fundus images of different retina images ,  20 ground truth vessel segmented images and 20 masks of retina fundus. In the test dataset also we have same configuration of images where we have to predict over 20 images.

AUC_ROC_SCORE for U-net model is 0.9805 and for ResU-Net model is 0.9812.

https://desireai.com/resu-net-retinal-vessel-segmentation/
