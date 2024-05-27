import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load pre-trained model (e.g., ResNet50)
model = tf.keras.applications.ResNet50(weights='imagenet')

# Preprocess image function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Display Grad-CAM
def grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Streamlit interface
def main():
    st.title("Image Classification App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        if st.button('Classify'):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)
            decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=5)[0]
            
            st.write("Predictions:")
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                st.write(f"{i + 1}: {label} ({score:.2f})")
            
            # Grad-CAM
            heatmap = grad_cam(model, img_array, 'conv5_block3_out')
            heatmap = cv2.resize(heatmap, (image.width, image.height))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
            
            st.image(superimposed_img, caption='Grad-CAM', use_column_width=True)
if __name__ == "__main__":
    main()