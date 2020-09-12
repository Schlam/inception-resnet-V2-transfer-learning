import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to data used for this model
DATA_PATH = '/content/drive/My Drive/caltech_birds2010/'
DATA_PATH_COLAB = '/content/drive/My Drive/caltech_birds2010/'
SUBSET = "training" # If subset == 'training', opt for the larger of both values below
SEED = 0 # Random seed for replicability

def get_data(subset, seed=SEED, verbose=True, **kwargs):
    """ 
    This function uses ImageDataGenerator 
    """
    total_images, batch_size = {
        "training":[5095,],
        "validation":[856*2,16]
    }.get(subset)

    # Generator to augment our training images
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.32,    
        # Allow for additional augmentation parameters
        **kwargs)
    flow = datagen.flow_from_directory(
        DATA_PATH,
        shuffle=False,
        target_size=(299,299),
        batch_size=batch_size,
        subset=subset,
        seed=seed)

    images = []
    labels = []
    # Iterate through our directory using .next()
    for i in range(total_images // batch_size):
        # Save the images and labels separately
        image_batch, label_batch = flow.next()
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

