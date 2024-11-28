# Study ImageNet, GoogleNet, and ResNet Convolutional Neural Networks

from tensorflow.keras.applications import ResNet50

# Load ResNet50 model pre-trained on ImageNet
model = ResNet50(weights='imagenet')

# Print model summary
model.summary()
