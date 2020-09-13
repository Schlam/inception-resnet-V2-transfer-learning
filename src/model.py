
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Dropout,Flatten



class NyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(4,activation=tf.nn.softmmax)

    def call(self,inputs)

# Optimizer and loss funciton
optimizer = tf.keras.optimizers.SGD(lr = 0.003)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# MaxPooling2D/AvgPooling2D as Inception model's output
pooling_type = 'avg'

# Dimension of bottom two layers 
layer1_width = 1200 
layer2_width = 300

# InceptionResNetV2 model for 
base_model = InceptionResNetV2(
    include_top=False,
    pooling=pooling_type
)

for layer in base_model.layers:

    # Set all model layers to untrainable
    base_model.trainable = False


### Trainable block
inputs = base_model.output
x = Dense(layer1_width, activation='relu', input_shape=(None,1536))(inputs)
x = Dropout(.2)(x)
x = Dense(layer2_width, activation='relu')(x)
x = Dropout(.2)(x)
outputs = Dense(200, activation='softmax')(x)


# Final classification model
model = Model(base_model.input,outputs)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
)

# model.summary()