import os
import numpy as np
import random


with open('sketchvideo_val.txt', 'r') as fd:
    imgs = [line.strip() for line in fd.readlines() if line.strip()]

num = len(imgs)
ref_num = 40

with open('sketchvideo_ref_test.txt', 'w') as fd:
    for i in range(num):
        fd.write(imgs[i])
        used = [i]
        for j in range(ref_num):
            idx = 0
            while 1:
                rd = random.random()
                idx = int(rd * num)
                if idx not in used:
                    used.append(idx)
                    break
            fd.write(',' + imgs[idx])
        fd.write('\n')
    