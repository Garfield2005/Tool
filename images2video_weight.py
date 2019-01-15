#/usr/bin/python

import os
import cv2
 
fps = 8.0
size = (1224,512) 
is_weight = True
weight_alpha = 0.6
weight_beta = 0.4

# 图片目录名称
image_set = "images_2019_01_15"
# 加权合成图片目录
image_source_dir = "/data/deeplearning/zhangmm/segmentation/images"
image_result_dir = "/data/deeplearning/zhangmm/segmentation/result"
# 抽取图片起止索引号
image_index_min = 0
image_index_max = 300
#是否截取原始图片的下半部分，作者个人使用场景原因才会有这个设置，只需要保证合成的两张图片size一致即可
crop_source_image = False
# 合成后的视频，其中一个是否需要延迟n张，作者个人使用场景决定的，正常设置为0即可
source_delay_frame = 0

image_source_dir = os.path.join(image_source_dir, image_set)
image_dir = os.path.join(image_result_dir, image_set)

# 目标视频存储目录
dest_video_dir = "/data/deeplearning/zhangmm/segmentation/output/video"
dest_video = os.path.join(dest_video_dir,image_set)
if source_delay_frame > 0:
   dest_video = dest_video + "_delay{}".format(source_delay_frame)
dest_video = dest_video + "_{}fps.avi".format(fps)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videowriter = cv2.VideoWriter(dest_video, fourcc, fps, size)

# 获取目录下图片信息
image_list = []
if os.path.isdir(image_dir):
    for fname in sorted(os.listdir(image_dir)):
        fpath = os.path.join(image_dir, fname)
        if fpath.endswith(".jpg") or fpath.endswith(".jpeg") or fpath.endswith(".png"):
            image_list.append(fpath)
        else:
            print("image file must be end with jpg/jpeg/png: {}".format(fpath))
else:
    image_list = image_dir

image_list = image_list[image_index_min:image_index_max]

# 根据情况将图片排序，因为决定了合成视频的顺序
#image_list.sort(key=lambda x:int(os.path.splitext(os.path.basename(x))[0]))
#image_list.sort(key=lambda x:int(os.path.splitext(os.path.basename(x).split("-frame-")[1])[0]))
print("get images: {}".format(len(image_list)))


for i, image_path in enumerate(image_list):
    print("{}: {}".format(i, image_path))
    result_image_path = image_path
    if source_delay_frame > 0 and (i - source_delay_frame) > 0:
       result_image_path = image_list[i - source_delay_frame]
       #print("  result: {}: {}".format(i + source_delay_frame, result_image_path))
    result_image = cv2.imread(result_image_path)
    source_image_path = os.path.join(image_source_dir, os.path.basename(image_path).split('_')[1])
    #print("  source: {}: {}".format(i, source_image_path))
    source_image = cv2.imread(source_image_path)
    if crop_source_image:
       h, w, _ = source_image.shape
       source_image = source_image[h//2:h, 0:w]
       source_image = cv2.resize(source_image, size, interpolation=cv2.INTER_LINEAR)
    dest_image = cv2.addWeighted(source_image, weight_alpha, result_image, weight_beta, 0)
    videowriter.write(dest_image)

videowriter.release()

# 将合成的avi视频转为mp4格式
cmdline = "ffmpeg -i " + dest_video + " -y " + dest_video + ".mp4"
print("avi to mp4: {}".format(cmdline))
os.system(cmdline)
