#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 700
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


# In[39]:


download_path = '/home/stmoon/Test/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz'
MODEL = DeepLabModel(download_path)


# In[31]:


import glob

dir_path = '/home/stmoon/Project/AE590/data/out4_20181102_170021/'
out_path = '/home/stmoon/Project/AE590/deeplab/out/out5'


def extract_road(input_path, output_path) :
    file_list = sorted(glob.glob(dir_path + '/*.png'))

    for f in file_list :
        img = Image.open(f) 

        resized_im, seg_map = MODEL.run(img)
        p = resized_im.load()

        for i in range(resized_im.size[0]) :
            for j in range(resized_im.size[1]) :
                if seg_map[j,i] == 0 :
                    p[i,j] = tuple([int(x*0.3) for x in p[i,j]])
        
        resized_im.save(out_path + '/' + f.split('/')[-1])
    


# In[69]:


import glob
import os.path

def extract_road(input_path, output_path) :
    file_list = sorted(glob.glob(input_path + '/*.png'))

    for f in file_list :
        file_path = output_path + '/' + f.split('/')[-1]
        file_path = file_path.replace('.png', '.txt')

        if os.path.exists(file_path) == False : 

            print(f)
 
            img = Image.open(f) 
            resized_im, seg_map = MODEL.run(img)
        
            np.savetxt(file_path, seg_map, fmt="%d", delimiter=",")
        
    


# In[70]:


import os

#20181102_163625.mp4  20181102_164146.mp4  20181102_164558.mp4  20181102_164952.mp4  20181102_165724.mp4
#20181102_163933.mp4  20181102_164357.mp4  20181102_164756.mp4  20181102_165208.mp4  20181102_170021.mp4

video_name = [
    '20181102_163625',  
    '20181102_164146',
    '20181102_164558',
    '20181102_164952',
    '20181102_165724',
    '20181102_163933',
    '20181102_164357',
    '20181102_164756',
    '20181102_165208',
    '20181102_170021']

for v in video_name :
    file_path = "/home/stmoon/Project/AE590/data/out_" + v
    out_path = "/home/stmoon/Project/AE590/deeplab/out/out_" + v
    print(file_path, out_path)
    os.makedirs(out_path, exist_ok=True)
    extract_road(file_path, out_path)


# In[ ]:


