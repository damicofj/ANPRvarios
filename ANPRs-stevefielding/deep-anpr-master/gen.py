#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



"""
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys
import re
import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (64, 128)

CHARS = common.CHARS + " "


def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color

# scale_variation seems to determine percentage of images that are considered out of bounds. Assume
# that rotation_variation and translation_variation do something similar
def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    # (0.6 + 0.875) * 0.5 - (0.875 - 0.6) * 0.5 * 1.5 ...
    # (0.7375 - 0.20625) to (0.7375 + 0.20625)
    # 0.53125 to 0.944
    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def generate_code():
    return "{}{}{}{}{}{}{}".format(
        random.choice(common.DIGITS),
        random.choice(common.LETTERS),
        random.choice(common.LETTERS),
        random.choice(common.LETTERS),
        random.choice(common.DIGITS),
        random.choice(common.DIGITS),
        random.choice(common.DIGITS))


def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.25, 0.35) * font_height
    v_padBot = random.uniform(0.15, 0.25) * font_height
    v_padTop = random.uniform(0.6, 0.7) * font_height

    # Note that the font already contains spacing, so it is possible to have negative spacing
    spacing = font_height * random.uniform(-0.065, -0.06)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padBot + v_padTop),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
    
    x = h_padding
    y = v_padTop
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(char_ims, num_bg_images):
    bg = generate_bg(num_bg_images)

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)
    plateShape = numpy.shape(plate)[::-1]
    plateShape = plateShape + (1,)
    plateShape = numpy.array([[0,0,1], plateShape, [0, plateShape[1], 1], [plateShape[0], 0, 1]])
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.3,
                            max_scale=0.8,
                            rotation_variation=0.4,
                            scale_variation=1.0,
                            translation_variation=1.0)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))
    # plate_box = numpy.array(M.dot(plateShape + (1,)))
    platePolygon = numpy.array((M.dot(plateShape.T)).T)
    #plate_box.shape = (2,2)
    platePolygon = platePolygon.astype(int)
    plate_box = numpy.array([numpy.min(platePolygon,axis=0), numpy.max(platePolygon,axis=0) ])

    # combine plate, and background. Use plate_mask to avoid plate and background corrupting each other
    # Then resize to target dimensions
    out = plate * plate_mask + bg * (1.0 - plate_mask)
    # cv2.rectangle(out, (plate_box[0,0],plate_box[0,1]), (plate_box[1,0],plate_box[1,1]), (0,255,0))
    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    # add some noise
    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code, plate_box, not out_of_bounds


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims


# create annotation text by replacing keywords within template text
def create_ann_text(jsonTemplateText, imageFileName,top,left,bottom,right):
  (x0,y0) = left, top
  (x1,y1) = right, top
  (x2,y2) = right, bottom
  (x3,y3) = left,  bottom
  width = right - left
  height = bottom - top
  annotation = re.sub(r'<x0>', str(x0), jsonTemplateText)
  annotation = re.sub(r"<x1>", str(x1), annotation)
  annotation = re.sub(r"<x2>", str(x2), annotation)
  annotation = re.sub(r"<x3>", str(x3), annotation)
  annotation = re.sub(r"<y0>", str(y0), annotation)
  annotation = re.sub(r"<y1>", str(y1), annotation)
  annotation = re.sub(r"<y2>", str(y2), annotation)
  annotation = re.sub(r"<y3>", str(y3), annotation)
  annotation = re.sub(r"<width>", str(width), annotation)
  annotation = re.sub(r"<height>", str(height), annotation)
  annotation = re.sub(r"<filename>", imageFileName, annotation)
  return annotation


def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    fonts, font_char_ims = load_fonts(FONT_DIR)
    num_bg_images = len(os.listdir("bgs"))
    while True:
        yield generate_im(font_char_ims[random.choice(fonts)], num_bg_images)


if __name__ == "__main__":
    plateOutputDir = "CA_artificial/img"
    annOutputDir = "CA_artificial/ann"
    bgOutputDir = "CA_artificial/bg"
    num_bg_images = len(os.listdir("bgs"))
    os.mkdir(plateOutputDir)
    os.mkdir(annOutputDir)
    os.mkdir(bgOutputDir)
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))
    numImageWithPlate = 0
    numImageNoPlate = 0
    jsonTemplateFile = open("plate_annotation_template.json", "r")
    jsonTemplateText = jsonTemplateFile.read()
    jsonTemplateFile.close()
    for img_idx, (im, c, plate_box, p) in enumerate(im_gen):
        imageFileName = "{:08d}_{}_{}.png".format(img_idx, c, "1" if p else "0")
        imageFileNameWithPath = "{}/{}".format(plateOutputDir, imageFileName)
        annFileName = re.sub(r".png", r".json", imageFileName)
        annFileNameWithPath = "{}/{}".format(annOutputDir, annFileName)
        print (imageFileName)
        annotation = create_ann_text(jsonTemplateText, imageFileName, plate_box[0,1], plate_box[0,0], plate_box[1,1], plate_box[1,0] )
        annFile = open(annFileNameWithPath, "w")
        annFile.write(annotation)
        annFile.close()
        cv2.imwrite(imageFileNameWithPath, im * 255.)
        if p:
          numImageWithPlate += 1
        else:
          numImageNoPlate += 1
    print("Images with plate: {}, images without plate: {}".format(numImageWithPlate, numImageNoPlate))
    for i in numpy.arange(int(sys.argv[1])):
      bg = generate_bg(num_bg_images)
      bgFileName = "{}/bg_{:08d}.png".format(bgOutputDir,i)
      cv2.imwrite(bgFileName, bg * 255.)
    print("Created {} background images...".format(int(sys.argv[1])))



