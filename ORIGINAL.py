# -*- coding: utf-8 -*-
"""caltech_birds2010_IRNV3_pooling_outputs

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1B0FJ9GBHDB-rXwf6TKKD-_Ls8N00JNA2



## Identifying bird species using InceptionResNetV2-based model
---


This repository uses data provided by CalTech (citaiton as follows)

@techreport{WelinderEtal2010,
	Author = {P. Welinder and S. Branson and T. Mita and C. Wah and F. Schroff and S. Belongie and P. Perona},
	Institution = {California Institute of Technology},
	Number = {CNS-TR-2010-001},
	Title = {{Caltech-UCSD Birds 200}},
	Year = {2010}
}


The *inception* architechture introduce in 2013 (?) by researchers at Google is excellent for classifying images where the subject is likely to appear in various sizes/positions in the image data.
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model,Input
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2






# Path to data used for this model
DATA_PATH = '/content/drive/My Drive/caltech_birds2010/'
DATA_PATH_COLAB = '/content/drive/My Drive/caltech_birds2010/'


SUBSET = "training" # If subset == 'training', opt for the larger of both values below
SEED = 0 # Random seed for replicability


def get_data(subset, seed=SEED, verbose=True, **kwargs):
    """
    
    This function uses ImageDataGenerator to produce data from google drive
    
    """
    total_images, batch_size = {

        "training":[5095,],
        "validation":[856*2,16]

    }.get(subset)


    # Generator to augment our training images
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.32,
        
        # Allow for additional augmentation parameters
        **kwargs 
    )

    flow = datagen.flow_from_directory(
        DATA_PATH,
        shuffle=False,
        target_size=(299,299),
        batch_size=batch_size,
        subset=subset,
        seed=seed
    )


    images = []
    labels = []

    # Iterate through our directory using .next()
    for i in range(total_images // batch_size):

        # Save the images and labels separately
        image_batch, label_batch = flow.next()

        # Convert from (batch_size, <# of classes>) to (batch_size,)
        # label_batch = [label.argmax for label in label_batch]

        # Add the images/labels from our data generator to a list
        images.extend(image_batch)
        labels.extend(label_batch)

        if verbose:
            if i % 10 == 0:
                print(f"Complete {i+1}/{total_images//batch_size}")

    print("Finished loading images from data generator")

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels



# Get test data
images, labels = get_data(subset="validation")

split_at = 1600

test_images = images[split_at:]
test_labels = labels[split_at:]

images = images[:split_at]
labels = labels[:split_at]

# Get training data
# images, labels = get_data("training", verbose=False)




# Show a few example images

plt.figure(figsize=(8,5),dpi=100)

example1, example2 = images[-1], images[-2]

ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

ax1.imshow(example1)
ax2.imshow(example2);

"""### $\S$ 1: Model architechture

Below is a graphic showing the model design which will serve as the base for our classifier:

[image](https://1.bp.blogspot.com/-O7AznVGY9js/V8cV_wKKsMI/AAAAAAAABKQ/maO7n2w3dT4Pkcmk7wgGqiSX5FUW2sfZgCLcB/s1600/image00.png)
"""



# Whether to use the 'max' pooling or 'avg' pooling layer as output
POOLING = 'avg'

tf.keras.backend.clear_session()

base_model = InceptionResNetV2(
    include_top=False,
    pooling=POOLING
)

print(f"Number of model layers: {len(base_model.layers)}")
print(f"Output shape: {base_model.output.shape}")



# Whether or not to truncate the base model further, and how deep/shallow
LITE = False
DEPTH = 2 # there are three options: 1,2, and 3


if LITE:

    output_layer = 'max_pooling2d_'+str(DEPTH)

    base_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(output_layer).output
    )

    print(f"Number of model layers: {len(base_model.layers)}")
    print(f"Output shape: {base_model.output.shape}")

# Inspect the shape of our resulting tensor

print("Shape of our dataset")
images.shape

# Use our model to predict on the image data

pooling_outputs = base_model.predict(
    images,
    verbose=1
)

print(f"Completed inference on dataset")

output_shape = pooling_outputs.shape
print(f"Shape of pooling_outputs: {output_shape}")

# Plot the model outputs if they aren't 1D tensors

if LITE:

    # Compare bird images to the sum of their pooling outputs
    BIRD = 12

    plt.figure(figsize=(8,5),dpi=100)

    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    ax1.imshow(images[BIRD])
    ax2.imshow(pooling_outputs[BIRD].sum(axis=-1))

"""#### Adding trainable layers to this model in order to fine-tune it's performance for our specific use case"""




# Freeze our base model so only the weights of our final layers can be trained
for layer in base_model.layers:
    base_model.trainable = False


# Create a block of dense/dropout layers to add to our dase
inputs = base_model.output
x = Dense(1200,activation='relu',input_shape=(None,1536))(inputs)
x = Dropout(.2)(x)
x = Dense(300,activation='relu')(x)
x = Dropout(.2)(x)
outputs = Dense(200,activation='softmax')(x)


# Final classification model
model = Model(base_model.input,outputs)

# model.summary()

# Set the specifications for any callback function you want to include

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2,
    patience=5, min_lr=0.001
)



# Chose the optimizer and loss function for this model

OPTIMIZER = tf.keras.optimizers.SGD(lr = 0.003)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()


# Compile the model
model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=['accuracy']
)

batch_size = 16
epochs = 2
shuffle = True

x_train = tf.data.Dataset.from_tensor_slices(images)
y_train = tf.data.Dataset.from_tensor_slices(labels)
x_test = tf.data.Dataset.from_tensor_slices(test_images)
y_test = tf.data.Dataset.from_tensor_slices(test_labels)


# Fit the model
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=shuffle,
    callbacks=[reduce_lr],
    validation_data=(x_test,y_test)
)
