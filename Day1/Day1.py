# Import `tensorflow`, os, skimage, and numpy
import tensorflow as tf
import os
import skimage
import numpy as np

# Import the `pyplot` module
import matplotlib.pyplot as plt

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Print the result
print(result)

#Belgian Traffic Signs

#Load Data
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/home/rj/Documents/Learning TensorFlow"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

# Print the `images` dimensions
# np.array() or numpy.array() needs to be used in order to convert the list to an array.
print(np.array(images).ndim)

# Print the number of `images`'s elements
# np.array() or numpy.array() needs to be used in order to convert the list to an array.
print(np.array(images).size)

# Print the first instance of `images`
images[0]

# Print the `labels` dimensions
# np.array() or numpy.array() needs to be used in order to convert the list to an array.
print(np.array(labels).ndim)

# Print the number of `labels`'s elements
# np.array() or numpy.array() needs to be used in order to convert the list to an array.
print(np.array(labels).size)

# Count the number of labels
print(len(set(labels)))

# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)

# Show the plot
#plt.show()

# Determine the (random) indexes of the images that you want to see 
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images that you defined 
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)

#plt.show()

# Determine the (random) indexes of the images
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images and add shape, min and max values
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    #plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
                                                  images[traffic_signs[i]].min(), 
                                                  images[traffic_signs[i]].max()))

# Get the unique labels 
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image)
    
# Show the plot
#plt.show()

# Rescale the images in the `images` array
# skimage was already imported in full
images28 = [skimage.transform.resize(image, (28, 28)) for image in images]

# Convert `images28` to an array
images28 = np.array(images28)

# Convert `images28` to grayscale
# skimage was already imported in full
images28 = skimage.color.rgb2gray(images28)

traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)
    
# Show the plot
plt.show()