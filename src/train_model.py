import tensorflow as tf
from model import Model
model = Model()


# Set the specifications for any callback function you want to include
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2,
    patience=5, min_lr=0.001
)


callbacks = [reduce_lr]
batch_size = 16
epochs = 2
shuffle_on_fit = True

# Fit the model
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle_on_fit=shuffle,
    callbacks=callbacks,
    validation_data=(x_test,y_test)
)
