import numpy as np
import cv2
import liblzfse
import os
from pathlib import Path

def save_rgb(img, episode_dir):
    img=img.reshape(480,640,3,-1)    
    #export img to vid
    
    out=cv2.VideoWriter((episode_dir / "compressed_video_h264.mp4").as_posix(), cv2.VideoWriter_fourcc(*'mp4v'), 10, (img.shape[1], img.shape[0]))
    for i in range(img.shape[-1]):
        out.write(cv2.cvtColor(img[:,:,:,i], cv2.COLOR_BGR2RGB)) #fix colors and write to vid
    out.release()

def save_depth(depth, episode_dir):
    depth=depth.reshape(480,640,1,-1)
    #export depth to zipped binary
    depth=depth.astype(np.float32)
    depth/=2000 #normalize to max out at ~2 meters
    depth=np.moveaxis(depth,-1,0)
    depth=np.squeeze(depth, axis=-1)
    print(depth.shape)
    depth_bytes=liblzfse.compress(depth.tobytes())
    (episode_dir / "compressed_np_depth_float32.bin").write_bytes(depth_bytes)

def process_episode(filename, root_dir):
    data=np.load(f'data/{filename}.npz')
    img=data["img"]
    depth=data["depth"]
    episode_dir_name = f'dataset/task1/env1/{filename}'
    try:
        os.mkdir(episode_dir_name)
    except FileExistsError:
        print("Folder already exists, continuing.")
    episode_dir = root_dir / episode_dir_name
    save_all(img, depth, episode_dir)   # save rgb and depth images
    os.rename(f'data/{filename}.json', episode_dir / 'labels.json')  # save action labels

def save_all(img, depth, episode_dir):
    save_rgb(img, episode_dir)
    with open(episode_dir / 'rgb_rel_videos_exported.txt', 'w') as f:
        f.write('Done.')
    save_depth(depth, episode_dir)
    with open(episode_dir / 'completed.txt', 'w') as f:
        f.write('Completed')

    

# read every .npz file in data/
datafiles=[]
for file in os.listdir("data"):
    if file.endswith(".npz"):
        datafiles.append(file.split(".")[0])

root_dir=Path(os.getcwd())
for file in datafiles:    
    process_episode(file, root_dir)
