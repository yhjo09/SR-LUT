
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# USER PARAMS
UPSCALE = 4                  # upscaling factor
MODEL_PATH = "./Model_S.pth"    # Trained SR net params
SAMPLING_INTERVAL = 4        # N bit uniform sampling






### A lightweight deep network ###
class SRNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(SRNet, self).__init__()

        self.upscale = upscale

        self.conv1 = nn.Conv2d(1, 64, [2,2], stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(64, 1*upscale*upscale, 1, stride=1, padding=0, dilation=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B*C, 1, H, W)

        x = self.conv1(x_in)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale*(H-1), self.upscale*(W-1))

        return x

model_G = SRNet(upscale=UPSCALE).cuda()



lm = torch.load('{}'.format(MODEL_PATH))
model_G.load_state_dict(lm.state_dict(), strict=True)



### Extract input-output pairs
with torch.no_grad():
    model_G.eval()

    # 1D input
    base = torch.arange(0, 257, 2**SAMPLING_INTERVAL)   # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.cuda().repeat(L)                             # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    third = base.cuda().unsqueeze(1).repeat(1, L*L).reshape(-1) # 256*256*256   0 x65536|1 x65536|...|255 x65536
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)    # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L*L*L).reshape(-1) # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)    # [256*256*256*256, 4]

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,1,2,2).float() / 255.0
    print("Input size: ", input_tensor.size())

    # Split input to not over GPU memory
    B = input_tensor.size(0) // 100
    outputs = []

    for b in range(100):
        if b == 99:
            batch_output = model_G(input_tensor[b*B:])
        else:
            batch_output = model_G(input_tensor[b*B:(b+1)*B])

        results = torch.round(torch.clamp(batch_output, -1, 1)*127).cpu().data.numpy().astype(np.int8)
        outputs += [ results ]
    
    results = np.concatenate(outputs, 0)
    print("Resulting LUT size: ", results.shape)

    np.save("Model_S_x{}_{}bit_int8".format(UPSCALE, SAMPLING_INTERVAL), results)





