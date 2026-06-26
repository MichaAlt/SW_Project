import tensorflow as tf

from tensorflow.keras import datasets, layers, models

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor='val_accuracy',

        patience=3,

        restore_best_weights=True

    )

]
def create_cnn():

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    model.compile(optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(
        train_images, 
        train_labels, 
        epochs=10,
        batch_size = 16,
        validation_data=(test_images, test_labels)
        )
    
    return model, history