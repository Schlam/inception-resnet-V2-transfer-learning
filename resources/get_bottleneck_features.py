from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import tensorflow as tf

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

