import os
import numpy as np
from skimage import io
from skimage.transform import resize


with open('sketchvideo_train.txt', 'r') as fd:
    image_files = fd.readlines()

total = len(image_files)
cnt = 0

# path/to/deepfashion directory
root = '/home/ubuntu/datasets/SketchVideo_img'
# path/to/save directory
save_root = '/home/ubuntu/datasets/SketchVideo_CoCos'

for image_file in image_files:
    image_file = image_file.replace('\\', '/').strip()
    if image_file == '':
        continue
    #image_file = image_file.replace('sketch', 'video').rsplit('.', 1)[0] + '.jpg'
    image_file = os.path.join(root, image_file)
    #print(image_file)
    image = io.imread(image_file)
    height = image.shape[0]
    width = image.shape[1]
    if width > height:
        pad_width_1 = (width - height) // 2
        pad_width_2 = width - height - pad_width_1
        image_pad = np.pad(image, ((pad_width_1, pad_width_2),(0,0),(0,0)), constant_values=232)
    else:
        pad_width_1 = (height - width) // 2
        pad_width_2 = height - width - pad_width_1
        image_pad = np.pad(image, ((0,0),(pad_width_1, pad_width_2),(0,0)), constant_values=232)
    image_resize = resize(image_pad, (512, 512))
    #image_resize = (image_resize * 255).astype('uint8')
    image_resize = (image_pad * 255).astype('uint8')
    dst_file = os.path.dirname(image_file).replace(root, save_root)
    os.makedirs(dst_file, exist_ok=True)
    dst_file = os.path.join(dst_file, os.path.basename(image_file))
    # dst_file = dst_file.replace('.jpg', '.png')
    io.imsave(dst_file, image_resize)
    cnt += 1
    if cnt % 20 == 0:
        print('Processing: %d / %d' % (cnt, total))
