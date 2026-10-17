import argparse

import numpy as np
from tensorflow import keras
from PIL import Image

import config


def parse_args():
    p = argparse.ArgumentParser(description="Classify a single waste image")
    p.add_argument("image", help="Path to an image file")
    p.add_argument("--model-path", default=str(config.MODEL_PATH))
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--top-k", type=int, default=6)
    return p.parse_args()


def load_and_preprocess(image_path, img_size):
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype="float32")
    return np.expand_dims(arr, axis=0)


def predict(model, image_path, img_size, class_names):
    batch = load_and_preprocess(image_path, img_size)
    probs = model.predict(batch, verbose=0)[0]
    return probs


def main():
    args = parse_args()
    model = keras.models.load_model(args.model_path)
    class_names = config.CLASS_NAMES

    probs = predict(model, args.image, args.img_size, class_names)
    order = np.argsort(probs)[::-1][: args.top_k]

    print(f"\nImage: {args.image}")
    print(f"Predicted: {class_names[order[0]]} ({probs[order[0]] * 100:.2f}%)\n")
    for idx in order:
        print(f"  {class_names[idx]:<10} {probs[idx] * 100:6.2f}%")


if __name__ == "__main__":
    main()
