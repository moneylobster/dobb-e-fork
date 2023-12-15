import numpy as np
import matplotlib.pyplot as plt
import cv2
import liblzfse
import os
from pathlib import Path

data=np.load("data/log.npz")


print(data.files)

img=data["img"]
depth=data["depth"]

img=img.reshape(480,640,3,-1)
print(img.shape)

depth=depth.reshape(480,640,1,-1)

#export img to vid
out=cv2.VideoWriter("data/compressed_video_h264.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (img.shape[1], img.shape[0]))
for i in range(img.shape[-1]):
    out.write(cv2.cvtColor(img[:,:,:,i], cv2.COLOR_BGR2RGB)) #fix colors and write to vid
out.release()

#export depth to zipped binary
depth=depth.astype(np.float32)
depth/=2000 #normalize to max out at ~2 meters
depth=np.moveaxis(depth,-1,0)
depth=np.squeeze(depth, axis=-1)
print(depth.shape)
depth_bytes=liblzfse.compress(depth.tobytes())

root_dir=Path(os.getcwd())
(root_dir / "compressed_np_depth_float32.bin").write_bytes(depth_bytes)


# for i in range(img.shape[-1]):
#     cv2.imshow("ha", img[:,:,:,i])
#     while True:
#         if cv2.waitKey(1) & 0xFF ==ord('q'):
#             break
#     cv2.destroyAllWindows()

# for i in range(depth.shape[-1]):
#     cv2.imshow("ha", depth[:,:,:,i])
#     while True:
#         if cv2.waitKey(1) & 0xFF ==ord('q'):
#             break
#     cv2.destroyAllWindows()
