import argparse
import config
import cv2
import os
import pickle
import PIL.Image
import numpy as np
import dnnlib.util as util
import dnnlib.tflib as tflib
from encoder.generator_model import Generator

"""
boundaries/       -input boundaries which name is <name>_boundary.npy
latents/          -input latent_codes
interpolations/   -output images folder 
"""


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Edit image from its latent vector with given semantic boundary.')
    parser.add_argument('-d', '--direction', required=False, type=str,
                        choices=['age', 'yaw', 'pitch', 'smile', 'eyeglasses', 'openmouth', 'gender'],
                        help='Direction to face turns.')
    parser.add_argument('-b', '--boundary_path', default='boundaries', required=False, type=str,
                        help='Path to the semantic boundary.')
    parser.add_argument('-i', '--input_path', default='latents', required=False, type=str,
                        help='Path to the latent codes.')
    parser.add_argument('-l', '--load_models', default='url', choices=['url', 'local'],
                        help='Fetch models  from predefined url')
    parser.add_argument('-n', '--number_interpolation_steps', default=9, type=int,
                        help='Number of interpolation steps')
    parser.add_argument('-s', '--morph_strength', default=2, type=int,
                        help='Morph`s strength in a boundary`s direction.')
    return parser.parse_args()


def preprocess(load: str, number_interpolation_steps: int, morph_strength: float):
    """
    Get generator and linspace for interpolations

    :param load: 'url'/'local' --from URL or local models/-directory
    :param number_interpolation_steps: steps (from -morph_strength to morph_strength)
    :param morph_strength: upper max boundary
    :return: generator, interpolation_steps
    """
    if load == 'url':
        with util.open_url('https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ',
                           cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)
    else:
        with open('models/pretrain/karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
    interpolation_steps = np.linspace(-morph_strength, morph_strength, number_interpolation_steps)
    return generator, interpolation_steps


def generate_image(latent_vector, generator):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img


def interpolate(latent_vector_name, boundary_name, latent_vector, boundary, coeffs, generator, show=False):
    folder = 'interpolations/' + str(latent_vector_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    folder = folder + '/' + str(boundary_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        NUM_LAYERS = 8  # played with it
        new_latent_vector[:NUM_LAYERS] = (latent_vector + coeff * boundary)[:NUM_LAYERS]
        # [:NUM_LAYERS, :] from (18,512) dlatent

        img = generate_image(new_latent_vector, generator)
        file_name = os.path.join(folder, str(i) + '.png')
        img.save(file_name)
        # img.show()
        if show:
            opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # from PIL to opencv
            cv2.imshow('next', opencv_image)
            cv2.waitKey()
    cv2.destroyAllWindows()


def extract_name(file_name: str, directory_name: str, delimiter='.'):
    """
    extract name from file_name before delimiter

    :param file_name:
    :param directory_name:
    :param delimiter:
    :return: extracted name, file_full_name
    """
    file_full_name = os.path.join(directory_name, file_name)
    idx = file_name.find(delimiter)
    name = file_name[:idx]
    return name, file_full_name


def main():
    args = parse_args()
    tflib.init_tf()
    generator, interpolation_steps = preprocess(args.load_models, args.number_interpolation_steps, args.morph_strength)

    if args.direction:
        boundary_files = [args.direction + '_boundary.npy']
    else:
        boundary_files = os.listdir(args.boundary_path)

    latent_vectors_directory = args.input_path
    latent_vectors_files = os.listdir(latent_vectors_directory)
    for f in latent_vectors_files:
        latent_vector_name, file_full_name = extract_name(f, latent_vectors_directory)
        latent_vector = np.load(file_full_name)
        if os.path.isdir(file_full_name):
            continue
        print('\n', f, ':')
        for ff in boundary_files:
            boundary_name, file_full_name = extract_name(ff, args.boundary_path, delimiter='_boundary')
            boundary = np.load(file_full_name)
            if os.path.isdir(file_full_name):
                continue
            print(boundary_name)
            interpolate(latent_vector_name, boundary_name, latent_vector, boundary, interpolation_steps, generator)
            # parameter show=True for display by steps


if __name__ == '__main__':
    main()
