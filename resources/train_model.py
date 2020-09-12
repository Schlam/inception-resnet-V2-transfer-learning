import tensorflow as tf



# Chose the optimizer and loss function for this model
CALLBACKS = [reduce_lr]
OPTIMIZER = tf.keras.optimizers.SGD(lr = 0.003)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
batch_size = 16
epochs = 2
shuffle_on_fit = True


# Set the specifications for any callback function you want to include
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2,
    patience=5, min_lr=0.001
)

# Compile the model
model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=['accuracy']
)

# Fit the model
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle_on_fit=shuffle,
    callbacks=CALLBACKS,
    validation_data=(x_test,y_test)
)
