# -*- coding: utf-8 -*-
# Copyright 2018-2020 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image marshalling."""

import io
import imghdr
import mimetypes

import numpy as np

from PIL import Image, ImageFile

from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from urllib.parse import urlparse

from streamlit.MediaFileManager import media_file_manager

LOGGER = get_logger(__name__)

# This constant is related to the frontend maximum content width specified
# in App.jsx main container
# 730 is the max width of element-container in the frontend, and 2x is for high
# DPI.
MAXIMUM_CONTENT_WIDTH = 2 * 730


def _image_has_alpha_channel(image):
    return image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )


def _PIL_to_bytes(image, format="JPEG", quality=100):
    format = format.upper()
    tmp = io.BytesIO()

    if _image_has_alpha_channel(image):
        image.save(tmp, format="PNG", quality=quality)
    else:
        image.save(tmp, format=format, quality=quality)

    return tmp.getvalue()


def _BytesIO_to_bytes(data):
    data.seek(0)
    return data.getvalue()


def _np_array_to_bytes(array, format="JPEG"):
    tmp = io.BytesIO()
    img = Image.fromarray(array.astype(np.uint8))

    if _image_has_alpha_channel(img):
        img.save(tmp, format="PNG")
    else:
        img.save(tmp, format=format)

    return tmp.getvalue()


def _4d_to_list_3d(array):
    return [array[i, :, :, :] for i in range(array.shape[0])]


def _verify_np_shape(array):
    if len(array.shape) not in (2, 3):
        raise StreamlitAPIException("Numpy shape has to be of length 2 or 3.")
    if len(array.shape) == 3 and array.shape[-1] not in (1, 3, 4):
        raise StreamlitAPIException(
            "Channel can only be 1, 3, or 4 got %d. Shape is %s"
            % (array.shape[-1], str(array.shape))
        )

    # If there's only one channel, convert is to x, y
    if len(array.shape) == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]

    return array


def _normalize_to_bytes(data, width, format):
    format = format.lower()
    ext = imghdr.what(None, data)

    if format is None:
        mimetype = mimetypes.guess_type(f"image.{ext}")[0]
    else:
        mimetype = f"image/{format}"

    image = Image.open(io.BytesIO(data))
    actual_width, actual_height = image.size

    if width < 0 and actual_width > MAXIMUM_CONTENT_WIDTH:
        width = MAXIMUM_CONTENT_WIDTH

    if width > 0 and actual_width > width:
        new_height = int(1.0 * actual_height * width / actual_width)
        image = image.resize((width, new_height))
        data = _PIL_to_bytes(image, format=format, quality=90)

        mimetype = "image/png" if format is None else f"image/{format}"
    return data, mimetype


def _clip_image(image, clamp):
    data = image
    if issubclass(image.dtype.type, np.floating):
        if clamp:
            data = np.clip(image, 0, 1.0)
        elif np.amin(image) < 0.0 or np.amax(image) > 1.0:
            raise RuntimeError("Data is outside [0.0, 1.0] and clamp is not set.")
        data = data * 255
    elif clamp:
        data = np.clip(image, 0, 255)
    elif np.amin(image) < 0 or np.amax(image) > 255:
        raise RuntimeError("Data is outside [0, 255] and clamp is not set.")
    return data


def marshall_images(
    image, caption, width, proto_imgs, clamp, channels="RGB", format="JPEG"
):
    channels = channels.upper()

    # Turn single image and caption into one element list.
    if type(image) is list:
        images = image
    elif type(image) == np.ndarray and len(image.shape) == 4:
        images = _4d_to_list_3d(image)
    else:
        images = [image]

    if type(caption) is list:
        captions = caption
    elif isinstance(caption, str):
        captions = [caption]
    elif type(caption) == np.ndarray and len(caption.shape) == 1:
        captions = caption.tolist()
    elif caption is None:
        captions = [None] * len(images)
    else:
        captions = [str(caption)]

    assert type(captions) == list, "If image is a list then caption should be as well"
    assert len(captions) == len(images), "Cannot pair %d captions with %d images." % (
        len(captions),
        len(images),
    )

    proto_imgs.width = width
    for image, caption in zip(images, captions):
        proto_img = proto_imgs.imgs.add()
        if caption is not None:
            proto_img.caption = str(caption)

        # PIL Images
        if isinstance(image, (ImageFile.ImageFile, Image.Image)):
            data = _PIL_to_bytes(image, format)

        elif type(image) is io.BytesIO:
            data = _BytesIO_to_bytes(image)

        elif type(image) is np.ndarray:
            data = _verify_np_shape(image)
            data = _clip_image(data, clamp)

            if channels == "BGR":
                if len(data.shape) == 3:
                    data = data[:, :, [2, 1, 0]]
                else:
                    raise StreamlitAPIException(
                        'When using `channels="BGR"`, the input image should '
                        "have exactly 3 color channels"
                    )

            data = _np_array_to_bytes(data, format=format)

        elif isinstance(image, str):
            # If it's a url, then set the protobuf and continue
            try:
                p = urlparse(image)
                if p.scheme:
                    proto_img.url = image
                    continue
            except UnicodeDecodeError:
                pass

            # If not, see if it's a file. Allow OS filesystem errors to raise.
            with open(image, "rb") as f:
                data = f.read()

        else:
            data = image

        (data, mimetype) = _normalize_to_bytes(data, width, format)
        this_file = media_file_manager.add(data, mimetype=mimetype)
        proto_img.url = this_file.url
