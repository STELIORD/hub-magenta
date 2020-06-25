import tensorflow as tf
import gradio
import matplotlib.pyplot as plt
import random
import os
from PIL import Image

def load():
    style_predict_path = tf.keras.utils.get_file('style_predict.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite')
    style_transform_path = tf.keras.utils.get_file('style_transform.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite')
    return style_predict_path, style_transform_path


def load_img(pil_img):
    filepath = "tmp/" + str(random.getrandbits(32)) + '.png'
    pil_img.save(filepath)
    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img, filepath


def preprocess_image(image, target_dim):
  # Resize the image so that the shorter dimension becomes 256px.
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
  return image



def run_style_predict(preprocessed_style_image, style_predict_path):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_predict_path)

    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

    return style_bottleneck


def run_style_transform(style_bottleneck, preprocessed_content_image,
                        style_transform_path):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_transform_path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Set model inputs.
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()

    # Transform content image.
    stylized_image = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()
    filepath = "tmp/" + str(random.getrandbits(32)) + '.png'
    if len(stylized_image.shape) > 3:
        stylized_image = tf.squeeze(stylized_image, axis=0)
        plt.imsave(filepath, stylized_image)
    else:
        plt.imsave(filepath, stylized_image)

    stylized_image = Image.open(filepath)
    os.remove(filepath)
    return stylized_image


def predict(content_img, style_img, models):
    style_predict_path, style_transform_path = models
    content_img, filepath_content = load_img(content_img)
    style_img, filepath_style = load_img(style_img)
    preprocessed_content_image = preprocess_image(content_img, 384)
    preprocessed_style_image = preprocess_image(style_img, 256)
    style_bottleneck = run_style_predict(preprocessed_style_image,
                                         style_predict_path)
    style_bottleneck_content = run_style_predict(
            preprocess_image(content_img, 256), style_predict_path)
    content_blending_ratio = 0.25
    style_bottleneck_blended = content_blending_ratio * \
                               style_bottleneck_content + \
                               (1 - content_blending_ratio) * \
                               style_bottleneck

    stylized_image_blended = run_style_transform(style_bottleneck_blended,
                                                 preprocessed_content_image,
                                                 style_transform_path)


    os.remove(filepath_content)
    os.remove(filepath_style)

    return stylized_image_blended


INPUTS = [gradio.inputs.ImageIn(cast_to="pillow"), gradio.inputs.ImageIn(
    cast_to="pillow")]
OUTPUTS = gradio.outputs.Image()
INTERFACE = gradio.Interface(fn=predict, inputs=INPUTS, outputs=OUTPUTS,
                             load_fn=load)

INTERFACE.launch(inbrowser=True)

#
# Function to load an image from a file, and add a batch dimension.
# def load_img(path_to_img):
#   img = tf.io.read_file(path_to_img)
#   img = tf.io.decode_image(img, channels=3)
#   img = tf.image.convert_image_dtype(img, tf.float32)
#   img = img[tf.newaxis, :]
#
#   return img

# Function to pre-process by resizing an central cropping it.

# Load the input images.
# content_image = load_img(content_path)
# style_image = load_img(style_path)

# Preprocess the input images.
# # preprocessed_content_image = preprocess_image(content_image, 384)
# # preprocessed_style_image = preprocess_image(style_image, 256)
#
# print('Style Image Shape:', preprocessed_style_image.shape)
# print('Content Image Shape:', preprocessed_content_image.shape)


# def imsave(file, image):
#     if len(image.shape) > 3:
#         image = tf.squeeze(image, axis=0)
#         plt.imsave(file, image)
#     else:
#         plt.imsave(file, image)


# imsave('results/preproc-content.png', preprocessed_content_image)
# imsave('results/preproc-style.png', preprocessed_style_image)

# Function to run style prediction on preprocessed style image.
# def run_style_predict(preprocessed_style_image):
#     # Load the model.
#     interpreter = tf.lite.Interpreter(model_path=style_predict_path)
#
#     # Set model input.
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)
#
#     # Calculate style bottleneck.
#     interpreter.invoke()
#     style_bottleneck = interpreter.tensor(
#       interpreter.get_output_details()[0]["index"]
#       )()
#
#     return style_bottleneck

# Calculate style bottleneck for the preprocessed style image.
# style_bottleneck = run_style_predict(preprocessed_style_image)
# print('Style Bottleneck Shape:', style_bottleneck.shape)

# Run style transform on preprocessed style image
# def run_style_transform(style_bottleneck, preprocessed_content_image):
#   # Load the model.
#   interpreter = tf.lite.Interpreter(model_path=style_transform_path)
#
#   # Set model input.
#   input_details = interpreter.get_input_details()
#   interpreter.allocate_tensors()
#
#   # Set model inputs.
#   interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
#   interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
#   interpreter.invoke()
#
#   # Transform content image.
#   stylized_image = interpreter.tensor(
#       interpreter.get_output_details()[0]["index"]
#       )()
#
#   return stylized_image

# Stylize the content image using the style bottleneck.
# stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image)

# Visualize the output.
# imsave('results/stylized-image.png', stylized_image)

# Calculate style bottleneck of the content image.
# style_bottleneck_content = run_style_predict(
#     preprocess_image(content_image, 256)
#     )

# Define content blending ratio between [0..1].
# 0.0: 0% style extracts from content image.
# 1.0: 100% style extracted from content image.
# content_blending_ratio = 0.5

# # Blend the style bottleneck of style image and content image
# style_bottleneck_blended = content_blending_ratio * style_bottleneck_content \
#                            + (1 - content_blending_ratio) * style_bottleneck
#
# # Stylize the content image using the style bottleneck.
# stylized_image_blended = run_style_transform(style_bottleneck_blended,
#                                              preprocessed_content_image)

# Visualize the output.
# imsave('results/stylized-image-blended.png', stylized_image_blended)
