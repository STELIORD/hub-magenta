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


def predict(content_img, style_img, content_blending_ratio, models):
    style_predict_path, style_transform_path = models
    content_img, filepath_content = load_img(content_img)
    style_img, filepath_style = load_img(style_img)
    preprocessed_content_image = preprocess_image(content_img, 384)
    preprocessed_style_image = preprocess_image(style_img, 256)
    style_bottleneck = run_style_predict(preprocessed_style_image,
                                         style_predict_path)
    style_bottleneck_content = run_style_predict(
            preprocess_image(content_img, 256), style_predict_path)
    # content_blending_ratio = 0.25
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


INPUTS = [gradio.inputs.ImageIn(cast_to="pillow", label="Content Image"),
          gradio.inputs.ImageIn(
    cast_to="pillow", label="Style Image"), gradio.inputs.Slider(0, 1,
                                                                 "Content "
                                                                 "Blending "
                                                                 "Ratio")]
OUTPUTS = gradio.outputs.Image(label="Stylized Image")

INTERFACE = gradio.Interface(fn=predict, inputs=INPUTS, outputs=OUTPUTS,
                             load_fn=load)

INTERFACE.launch(inbrowser=True, share=True)
