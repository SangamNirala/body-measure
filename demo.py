"""
Demo of HMR (Human Mesh Recovery).

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when the max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply the output of OpenPose to figure out the bounding box and the right scale factor.

Sample usage:
# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with OpenPose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import extract_measurements
import sys
import cv2
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

# Define command-line flags for input image and OpenPose JSON path
flags.DEFINE_string('img_path', 'data/k3.png', 'Image to run')
flags.DEFINE_string('json_path', None, 'If specified, uses the OpenPose output to crop the image.')

def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in the original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()

def preprocess_image(img_path, json_path=None):
    """
    Preprocesses the input image and prepares it for the model.
    
    Args:
        img_path: Path to the input image.
        json_path: Optional path to OpenPose output for cropping.

    Returns:
        crop: Preprocessed image.
        proc_param: Processing parameters.
        img: Original image.
    """
    img = img_path  # Load the image
    if img.shape[2] == 4:
        img = img[:, :, :3]  # Remove alpha channel if present

    if json_path is None:
        if np.max(img.shape[:2]) != 224:
            scale = (float(224) / np.max(img.shape[:2]))  # Calculate scale
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)  # Calculate center
        center = center[::-1]  # Reverse for (x, y) format
    else:
        scale, center = op_util.get_bbox(json_path)  # Get bounding box from OpenPose

    crop, proc_param = img_util.scale_and_crop(img, scale, center, 224)  # Scale and crop the image

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

def main(img_path, height, json_path=None):
    """
    Main function to run the model on the input image.
    
    Args:
        img_path: Path to the input image.
        height: Height of the person in the image.
        json_path: Optional path to OpenPose output for cropping.
    """
    sess = tf.Session()  # Create a TensorFlow session
    model = RunModel(sess=sess)  # Initialize the model

    input_img, proc_param, img = preprocess_image(img_path, json_path)  # Preprocess the image
    input_img = np.expand_dims(input_img, 0)  # Add batch dimension

    # Run the model to predict joints, vertices, and camera parameters
    joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)

    # Extract measurements from the vertices
    extract_measurements.extract_measurements(height, verts[0])

if __name__ == '__main__':
    config = flags.FLAGS  # Get the configuration flags
    main(config)
```

Next, I will read the `extract_measurements.py` file to continue adding comments. 

<read_file>
<path>extract_measurements.py</path>
</read_file>
