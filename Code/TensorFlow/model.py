import tensorflow as tf
import os
def load_data(data_dir):
    batch_size = 32
    img_height = 3000
    img_width = 4000
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

def model_build():
    num_classes = 3

    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(80, activation='relu'),
        tf.keras.layers.Dense(70, activation='relu'),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model

def train(model, train_ds, val_ds):
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )
    return model

def main(data_dir=r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\test"):
    train_ds, val_ds = load_data(data_dir)
    model = model_build()
    model = train(model=model,train_ds=train_ds, val_ds=val_ds)
    model.save(os.path.join(r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\TensorFlow", "model"))
    
if __name__ == '__main__':
    main()