import argparse, json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np 
from PIL import Image
from config import IMAGE_SIZE

parser = argparse.ArgumentParser(description="Simple image classification script.")

parser.add_argument('image_path', action='store', type=str, help='path to the flower image')
parser.add_argument('model_path', action='store', type=str, help='path to saved model (.h5)')
parser.add_argument('--top_k', action='store', type=int, default=3, help='number of top probabilities to display')
parser.add_argument('--category_names', action='store', type=str, default='./flower_mapping.json', help='dictionary for mapping labels (json file)')

args = parser.parse_args()

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    image = np.asarray(Image.open(image_path))
    processed = process_image(image)
    probs = model.predict(np.expand_dims(processed, axis=0))[0]
    indices = (-probs).argsort()[:top_k]
    return list(probs[indices]), [str(idx+1) for idx in indices]

if __name__ == "__main__":
    # CUDA only
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print('No GPU detected')
        
    # load saved model
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    # calculate probabilities
    probs, classes = predict(args.image_path, model, args.top_k)

    # read and map class_names
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    flower_names = [ class_names[label] for label in classes ]

    print(f'\nTop {args.top_k} probabilities for image {args.image_path}:')
    for prob, flower in zip(probs, flower_names):
        print(f'{prob:.6f} - {flower}')
