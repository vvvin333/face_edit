import argparse
import cv2
import os
import pickle
import PIL.Image
import numpy as np

import config
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
    parser.add_argument('-a', '--axis', required=False, choices=['yaw', 'pitch'], type=str,
                        help='Direction to face turns.')
    parser.add_argument('-b', '--boundary_path', default='boundaries', required=False, type=str,
                        help='Path to the semantic boundary.')
    parser.add_argument('-i', '--input_latent_codes_path', default='latents', required=False, type=str,
                        help='Path to the latent codes.')
    parser.add_argument('-m', '--models', default='url', choices=['url', 'local'],
                        help='Fetch models  from predefined url')
    parser.add_argument('-n', '--number_interpolation_steps', default=9, type=int,
                        help='Number of interpolation steps')
    parser.add_argument('-s', '--morph_strength', default=2, type=int,
                        help='Morph`s strength in a boundary`s direction.')
    return parser.parse_args()


def generate_image(latent_vector, generator):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img


def interpolate(latent_vector_name, direction_name, latent_vector, direction, coeffs, generator, show=False):
    folder = 'interpolations/' + str(latent_vector_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    folder = folder + '/' + str(direction_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff * direction)[:8]  # [:8, :] from (18,512) dlatent
        img = generate_image(new_latent_vector, generator)
        file_name = os.path.join(folder, str(i) + '.png')
        img.save(file_name)
        # img.show()
        if show:
            opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # from PIL to opencv
            cv2.imshow('next', opencv_image)
            cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    tflib.init_tf()
    if args.models == 'url':
        with util.open_url('https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ',
                           cache_dir=config.cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)
    else:
        with open('models/karras2019stylegan-ffhq-1024x1024.pkl', 'rb') as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
    number_interpolation_steps = args.number_interpolation_steps
    morph_strength = args.morph_strength
    interpolation_steps = np.linspace(-morph_strength, morph_strength, number_interpolation_steps)

    if args.axis:
        boundary_files = [args.axis + '_boundary.npy']
    else:
        boundary_files = os.listdir(args.boundary_path)

    for f in os.listdir(args.input_latent_codes_path):
        file_full_name = os.path.join(args.input_latent_codes_path, f)
        if os.path.isdir(file_full_name):
            continue
        print('\n', f, ':')
        idx = f.find('.')
        latent_vector_name = f[:idx]
        latent_vector = np.load(file_full_name)
        for ff in boundary_files:
            file_full_name = os.path.join(args.boundary_path, ff)
            if os.path.isdir(file_full_name):
                continue
            idx = ff.find('_boundary')
            direction_name = ff[:idx]
            print(direction_name)
            boundary = np.load(file_full_name)
            interpolate(latent_vector_name, direction_name, latent_vector, boundary, interpolation_steps, generator)
            # show=True for display by steps


if __name__ == '__main__':
    main()
