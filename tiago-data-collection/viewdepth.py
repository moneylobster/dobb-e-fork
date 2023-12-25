import os
from pathlib import Path
import liblzfse
import numpy as np

#import skimage.io as io
import cv2

epname=input("Give the episode name date thing pls: ")
root_dir=Path(os.getcwd())
depth_file=root_dir / f"dataset/task1/env1/{epname}/compressed_np_depth_float32.bin"
raw_bytes=depth_file.read_bytes()
decompressed_bytes = liblzfse.decompress(raw_bytes)
depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
depth_img = depth_img.reshape((-1, 480, 640))
print(depth_img.shape)
print(np.max(depth_img))
#depth_img = np.ascontiguousarray(np.rot90(depth_img, -1))
#dimg=cv2.normalize(depth_img[len(depth_img)//2], None, 0,255, cv2.NORM_MINMAX)
#print(np.max(dimg))
dimg=depth_img[len(depth_img)//2]
cv2.imshow("a",dimg)
while True:
    if cv2.waitKey(0) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()
