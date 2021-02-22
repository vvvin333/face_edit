import cv2
import numpy as np
import os.path
import torch
from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator


# from utils.manipulator import linear_interpolate


def build_generator(model_name):
    """Builds the generator by model name."""
    gan_type = MODEL_POOL[model_name]['gan_type']
    if gan_type == 'pggan':
        generator = PGGANGenerator(model_name)
    elif gan_type == 'stylegan':
        generator = StyleGANGenerator(model_name)
    return generator


def sample_codes(generator, num, latent_space_type='Z', seed=0):
    """Samples latent codes randomly."""
    np.random.seed(seed)
    codes = generator.easy_sample(num)
    if generator.gan_type == 'stylegan' and latent_space_type == 'W':
        codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
        codes = generator.get_value(generator.model.mapping(codes))
    return codes


def interpolate(latent_codes, boundary, coeff: float, generator, synthesis_kwargs: dict):
    new_codes = latent_codes.copy()
    new_codes += boundary * coeff
    new_images = generator.easy_synthesize(new_codes, **synthesis_kwargs)['image']
    return new_images


def image_processing(images, col: int, viz_size=256):
    """
    processing images to one figure.
    :returns np.array(dtype=np.uint8) with RGB2BGR channels
    """
    num, height, width, channels = images.shape
    assert num % col == 0
    row = num // col
    fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)
    for idx, image in enumerate(images):
        i, j = divmod(idx, col)
        y = i * viz_size
        x = j * viz_size
        if height != viz_size or width != viz_size:
            image = cv2.resize(image, (viz_size, viz_size))
        fused_image[y:y + viz_size, x:x + viz_size] = image
    fused_image = np.asarray(fused_image, dtype=np.uint8)
    fused_image = cv2.cvtColor(fused_image, cv2.COLOR_RGB2BGR)
    return fused_image


def generate_sample(model_name="stylegan_ffhq", images_output_directory='generated_images', latent_space_type="W",
                    num_samples=1, noise_seed=111):
    """

    :param model_name:
    :param images_output_directory:
    :param latent_space_type:
    :param num_samples:
    :param noise_seed:
    :return:
    """
    generator = build_generator(model_name)
    latent_codes = sample_codes(generator, num_samples, latent_space_type, noise_seed)
    if generator.gan_type == 'stylegan' and latent_space_type.upper() == 'W':
        synthesis_kwargs = {'latent_space_type': 'W'}
    else:
        synthesis_kwargs = {}

    latent_file_name = 'sample.npy'
    image_file_name = 'base_.jpeg'
    generating_common_final(latent_codes, image_file_name, latent_file_name, generator, images_output_directory,
                            num_samples, synthesis_kwargs)

    return latent_codes, generator, synthesis_kwargs


def manipulate_with_params(latent_input_file: str, directions, model_name="stylegan_ffhq",
                           images_output_directory='generated_images', latent_space_type="W"):
    """

    :param latent_input_file:
    :param directions:
    :param model_name:
    :param images_output_directory:
    :param latent_space_type:
    :return:
    """
    generator = build_generator(model_name)
    if generator.gan_type == 'stylegan' and latent_space_type.upper() == 'W':
        synthesis_kwargs = {'latent_space_type': 'W'}
    else:
        synthesis_kwargs = {}

    latent_codes = np.load(latent_input_file)
    num_samples = latent_codes.shape[0]
    print(num_samples, ' samples loaded')

    boundaries = {}
    for attr_name in directions.keys():
        boundary_name = f'{model_name}_{attr_name}'
        if generator.gan_type == 'stylegan' and latent_space_type == 'W':
            boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_w_boundary.npy')
        else:
            boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_boundary.npy')

    for attr_name, attr_value in directions.items():
        latent_codes += boundaries[attr_name] * attr_value

    latent_file_name = 'sample_with_params.npy'
    image_file_name = 'sample_with_params.jpeg'
    generating_common_final(latent_codes, image_file_name, latent_file_name, generator, images_output_directory,
                            num_samples, synthesis_kwargs)

    return latent_codes, generator, synthesis_kwargs


def generating_common_final(latent_codes, image_file_name, latent_file_name, generator, images_output_directory,
                            num_samples, synthesis_kwargs):
    file_name = os.path.join('latents', latent_file_name)
    np.save(file_name, latent_codes)
    print(file_name, 'saved')

    images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']
    images = image_processing(images, num_samples)

    file_name = os.path.join(images_output_directory, image_file_name)
    cv2.imwrite(file_name, images)
    print(file_name, 'saved')
    cv2.imshow(image_file_name, images)
    cv2.waitKey()
    cv2.destroyAllWindows()
