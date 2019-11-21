# CycleGAN-CBCT-to-CT
Convert CBCT images to CT like images
This is 2D CycleGAN model.
Before training, resampling your CBCT's and CT's voxel spacing to the same voxel spacing. For our project, we resampled CT's voxel spacing to CBCT's voxel spacing, which is 0.51mm*0.51mm*1.99mm and then corpped to 512*512 dimensions. 
Input image size is 512*512 for both CBCT and CT images, and input image/data is in HU unit. Data normalization is done in train_cyclegan_patchD.py.
models_cyclegan_patchD.py includes CycleGAN model.
train_cyclegan_patchD.py includes training.
deploy_cyclegan_patchD.py includes testing
config_cyclegan_patchD includes config parameters.

Related paper: Generating synthesized computed tomography (CT) from cone-beam computed tomography (CBCT) using CycleGAN for adaptive radiation therapy. DOI: 10.1088/1361-6560/ab22f9. PMB

Link: https://www.ncbi.nlm.nih.gov/pubmed/31108465

Contact: xiao.liang@UTSouthwestern.edu
