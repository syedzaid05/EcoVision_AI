from tensorflow import keras

import config


def build_model(
    num_classes=config.NUM_CLASSES,
    img_size=config.IMG_SIZE,
    dropout=config.DROPOUT,
    trainable_base=False,
):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")

    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)

    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="waste_mobilenetv2")
    return model, base


def unfreeze_from(base, fine_tune_at=config.FINE_TUNE_AT):
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base.layers[fine_tune_at:]:
        layer.trainable = True


def compile_model(model, learning_rate, metrics=None):
    if metrics is None:
        metrics = ["accuracy"]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=metrics,
    )
    return model
