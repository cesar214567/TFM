import math
import os
import numpy as np
import cv2
#function fo sample the frames (there may be repeated depending on the total frames)
def sample_frames(frames, samples):
    x = len(frames)
    if x == 0 or samples <= 0:
        raise ValueError("frames must not be empty and y must be > 0")
    
    step = x / samples
    sampled_frames = [frames[min(math.floor(i * step), x - 1)] for i in range(samples)]
    return np.array(sampled_frames)

# actual function to get the frames using cv2
def get_frames(current_dir, file_name,img_size, images_per_file):
    in_file = os.path.join(current_dir, file_name)
    images = []

    vidcap = cv2.VideoCapture(in_file)
    success, image = vidcap.read()

    while success:
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = cv2.resize(RGB_img, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
        images.append(res)

        success, image = vidcap.read()
        
    resul = np.array(images)
    resul = (resul).astype(np.float16)
    resul = sample_frames(resul, images_per_file)

    return resul
