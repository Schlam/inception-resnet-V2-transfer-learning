
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import Dense,Dropout,Flatten



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