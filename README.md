## Practical Single-Image Super-Resolution Using Look-Up Table

[[Paper]]() 


## Dependency
- Python 3.6
- PyTorch 
- glob
- numpy
- pillow
- tqdm
- tensorboardx


## 1. Training deep SR network
1. Clone this repo.
```
git clone https://github.com/yhjo09/SR-LUT
```

2. Prepare DIV2K training images into `./1_Train_deep_model/train`.
- HR images should be placed as `./1_Train_deep_model/train/DIV2K_train_HR/*.png`.
- LR images should be placed as `./1_Train_deep_model/train/DIV2K_train_LR_bicubic/X4/*.png`.


3. Set5 HR/LR validation png images are already included in `./1_Train_deep_model/val`, or can use other images.

4. You may modify user parameters in L22 in `./1_Train_deep_model/Train_Model_S.py`.

5. Run.
```
cd ./1_Train_deep_model
python Train_Model_S.py
```

6. Checkpoints will be saved in `./1_Train_deep_model/SR-LUT/checkpoint/S`.



## 2. Transferring to LUT
1. Modify user parameters in L11 in `./2_Transfer_to_LUT/Transfer_Model_S.py` to 
- Specify a saved checkpoint in the step 1, or use attached `./2_Transfer_to_LUT/Model_S.pth`.

2. Run.
```
cd ./2_Transfer_to_LUT
python Transfer_Model_S.py
```

3. The resulting LUT will be saved like `./2_Transfer_to_LUT/Model_S_x4_4bit_int8.npy`.


## 3. Testing using LUT
1. Modify user parameters in L18 in `./3_Test_using_LUT/Test_Model_S.py`.
- Specify the generated LUT in the step 2, or use attached LUTs (npy files).

2. Set5 HR/LR test images are already included in `./3_Test_using_LUT/test`, or can use other images.

3. Run.
```
cd ./3_Test_using_LUT
python Test_Model_S.py (or Test_Model_F.py or Test_Model_V.py)
```

4. Resulting images will be saved in `./output_S_x4_4bit/*.png`.

5. We can reproduce the results of Table 6 in the paper, by modifying the variable `SAMPLING_INTERVAL` in L19 in Test_Model_S.py to range 3-8.



## 4. Testing on a smartphone
1. Download [SR-LUT.apk](https://drive.google.com/file/d/1Od4uoMeM6ND26yvKAsT3ofzwIDtO0LSn/view?usp=sharing) and install it.

2. You can test Set14 images or your own images.

![SR-LUT Android app demo](Demo.jpg)




## BibTeX
```
@InProceedings{jo2021practical,
   author = {Jo, Younghyun and Kim, Seon Joo},
   title = {Practical Single-Image Super-Resolution Using Look-Up Table},
   booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month = {June},
   year = {2021}
}
```

