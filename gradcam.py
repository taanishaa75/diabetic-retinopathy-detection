import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("dr_model.h5")

def make_heatmap(img_path):

    img = cv2.imread(img_path)
    img = cv2.resize(img,(224,224))
    img_array = np.expand_dims(img/255.0, axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("Conv_1").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)

    heatmap = tf.reduce_mean(grads, axis=(0,1,2))

    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)

    return heatmap.numpy()