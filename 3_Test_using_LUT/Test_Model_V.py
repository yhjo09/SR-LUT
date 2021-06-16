
from PIL import Image
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import glob
from tqdm import tqdm


import sys
sys.path.insert(1, '../1_Train_deep_model')
from utils import PSNR, _rgb2ycbcr



# USER PARAMS
UPSCALE = 4     # upscaling factor
SAMPLING_INTERVAL = 0        # N bit uniform sampling
LUT_PATH = "Model_V_x{}_{}bit_int8.npy".format(UPSCALE, SAMPLING_INTERVAL)    # Trained SR net params
TEST_DIR = './test/'      # Test images



# Load LUT
LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)



# Test LR images
files_lr = glob.glob(TEST_DIR + '/LR_x{}/*.png'.format(UPSCALE))
files_lr.sort()

# Test GT images
files_gt = glob.glob(TEST_DIR + '/HR/*.png')
files_gt.sort()


psnrs = []

if not isdir('./output_V_x{}_{}bit'.format(UPSCALE, SAMPLING_INTERVAL)):
    mkdir('./output_V_x{}_{}bit'.format(UPSCALE, SAMPLING_INTERVAL))

for ti, fn in enumerate(tqdm(files_gt)):
    # Load LR image
    img_lr = np.array(Image.open(files_lr[ti])).astype(np.float32)
    h, w, c = img_lr.shape

    # Load GT image
    img_gt = np.array(Image.open(files_gt[ti]))
    

    # Sampling interval for input
    q = 2**SAMPLING_INTERVAL


    # 3D equivalent of triangular interpolation
    def TriangularInterp(weight, img_in, h, w, q, rot, upscale=4):
        L = 2**(8-SAMPLING_INTERVAL) + 1

        # Extract MSBs
        img_a1 = img_in[:, 0:0+h, 0:0+w] // q
        img_b1 = img_in[:, 0:0+h, 1:1+w] // q
        
        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1

        # Extract LSBs
        fa_ = img_in[:, 0:0+h, 0:0+w] % q
        fb_ = img_in[:, 0:0+h, 1:1+w] % q


        # Vertices (O in Eq3 and Table3 in the paper)
        p00 = weight[ img_a1.flatten().astype(np.int_)*L + img_b1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p01 = weight[ img_a1.flatten().astype(np.int_)*L + img_b2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        
        p10 = weight[ img_a2.flatten().astype(np.int_)*L + img_b1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        p11 = weight[ img_a2.flatten().astype(np.int_)*L + img_b2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
        
        # Output image holder
        out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

        # Naive pixelwise output value interpolation 
        # It would be faster implemented with a parallel operation
        for c in range(img_a1.shape[0]):
            for y in range(img_a1.shape[1]):
                for x in range(img_a1.shape[2]):
                    fx = fa_[c,y,x]
                    fy = fb_[c,y,x]

                    if fx > fy:
                        out[c,y,x] = (q-fx) * p00[c,y,x] + (fx-fy) * p10[c,y,x] + (fy) * p11[c,y,x]
                    else:
                        out[c,y,x] = (q-fy) * p00[c,y,x] + (fy-fx) * p01[c,y,x] + (fx) * p11[c,y,x]

        out = np.transpose(out, (0, 1,3, 2,4)).reshape((img_a1.shape[0], img_a1.shape[1]*upscale, img_a1.shape[2]*upscale))
        out = np.rot90(out, rot, [1,2])
        out = out / q
        return out
    
    
    # Rotational ensemble
    img_in = np.pad(img_lr, ((0,0), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r0 = TriangularInterp(LUT, img_in, h, w, q, 0, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 1), ((0,0), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r1 = TriangularInterp(LUT, img_in, w, h, q, 3, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 2), ((0,0), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r2 = TriangularInterp(LUT, img_in, h, w, q, 2, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 3), ((0,0), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r3 = TriangularInterp(LUT, img_in, w, h, q, 1, upscale=UPSCALE)

    img_out = (out_r0/1.0 + out_r1/1.0 + out_r2/1.0 + out_r3/1.0) / 255.0
    img_out = img_out.transpose((1,2,0))
    img_out = np.round(np.clip(img_out, 0, 1) * 255).astype(np.uint8)

    # Matching image sizes 
    if img_gt.shape[0] < img_out.shape[0]:
        img_target = img_out[:img_gt.shape[0]]
    if img_gt.shape[1] < img_out.shape[1]:
        img_target = img_out[:, :img_gt.shape[1]]

    if img_gt.shape[0] > img_out.shape[0]:
        img_target = np.pad(img_out, ((0,img_gt.shape[0]-img_out.shape[0]),(0,0),(0,0)))
    if img_gt.shape[1] > img_out.shape[1]:
        img_target = np.pad(img_out, ((0,0),(0,img_gt.shape[1]-img_out.shape[1]),(0,0)))

    # Save to file
    Image.fromarray(img_out).save('./output_V_x{}_{}bit/{}_LUT_interp_{}bit.png'.format(UPSCALE, SAMPLING_INTERVAL, fn.split('/')[-1][:-4], SAMPLING_INTERVAL))

    CROP_S = 4
    psnr = PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(img_out)[:,:,0], CROP_S)
    psnrs.append(psnr)

print('AVG PSNR: {}'.format(np.mean(np.asarray(psnrs))))

