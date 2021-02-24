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
    return new_images, new_codes


def image_processing(images, col: int, viz_size=1024):
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
                    num_samples=1, noise_seed=111, resolution=1024):
    """

    :param resolution:
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

    latent_file_name = 'sample_base.npy'
    image_file_name = 'sample_base.jpeg'
    generating_common_final(latent_codes, image_file_name, latent_file_name, generator, images_output_directory,
                            num_samples, resolution, synthesis_kwargs)

    return latent_codes, generator, synthesis_kwargs


def linear_interpolations(latent_input_file: str, directions, num_steps=10, model_name="stylegan_ffhq",
                          images_output_directory='generated_images', latent_space_type="W", resolution=1024,
                          show=True):
    boundaries, generator, latent_codes, num_samples, synthesis_kwargs = \
        interpolation_common_preload(latent_input_file, directions, latent_space_type, model_name)

    channels = 3
    images_batch = np.zeros((num_steps * resolution, num_samples * resolution, channels), dtype=np.uint8)
    for attr_name, attr_value in directions.items():
        if attr_value == 0:
            continue

        left_boundary = - abs(attr_value)
        delta = abs(attr_value) * 2

        output_directory = os.path.join(images_output_directory, attr_name)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for step in range(num_steps):
            coeff = left_boundary + step * delta / num_steps
            images, _ = interpolate(latent_codes, boundaries[attr_name], coeff, generator, synthesis_kwargs)
            images = image_processing(images, num_samples, resolution)
            file_name = os.path.join(output_directory, attr_name + '_' + str(step) + '.jpeg')
            cv2.imwrite(file_name, images)
            print(file_name, 'saved')
            y = step * resolution
            if images.shape[0] != resolution:
                images = cv2.resize(images, (resolution * num_samples, resolution))  # (width, height)
            images_batch[y:y + resolution, :] = np.asarray(images, dtype=np.uint8)

        # images_batch = cv2.cvtColor(images_batch, cv2.COLOR_RGB2BGR)
        file_name = os.path.join(output_directory, attr_name + '_' + 'interpolations.jpeg')
        cv2.imwrite(file_name, images_batch)
        print(file_name, 'saved')
        if show:
            cv2.imshow(file_name, images_batch)
            cv2.waitKey()
            cv2.destroyAllWindows()


def manipulate_with_params(latent_input_file: str, directions, model_name='stylegan_ffhq',
                           images_output_directory='generated_images', latent_space_type='W', resolution=1024):
    """

    :param resolution:
    :param latent_input_file:
    :param directions:
    :param model_name:
    :param images_output_directory:
    :param latent_space_type:
    :return:
    """
    boundaries, generator, latent_codes, num_samples, synthesis_kwargs = \
        interpolation_common_preload(latent_input_file, directions, latent_space_type, model_name)

    for attr_name, attr_value in directions.items():
        latent_codes += boundaries[attr_name] * attr_value

    latent_file_name = 'sample_with_params.npy'
    image_file_name = 'sample_with_params.jpeg'
    generating_common_final(latent_codes, image_file_name, latent_file_name, generator, images_output_directory,
                            num_samples, resolution, synthesis_kwargs)

    return latent_codes, generator, synthesis_kwargs


def generating_common_final(latent_codes, image_file_name, latent_file_name, generator, images_output_directory,
                            num_samples, resolution, synthesis_kwargs):
    file_name = os.path.join('latents', latent_file_name)
    np.save(file_name, latent_codes)
    print(file_name, 'saved')

    images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']
    images = image_processing(images, num_samples, resolution)

    file_name = os.path.join(images_output_directory, image_file_name)
    cv2.imwrite(file_name, images)
    print(file_name, 'saved')
    cv2.imshow(image_file_name, cv2.resize(images, (num_samples*300, 300)))
    cv2.waitKey()
    cv2.destroyAllWindows()


def interpolation_common_preload(latent_input_file, directions, latent_space_type='W', model_name='stylegan_ffhq'):
    """

    :param directions:
    :param latent_input_file:
    :param latent_space_type:
    :param model_name:
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
    for attr_name in directions:
        boundary_name = f'{model_name}_{attr_name}'
        if generator.gan_type == 'stylegan' and latent_space_type.upper() == 'W':
            boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_w_boundary.npy')
        else:
            boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_boundary.npy')

    return boundaries, generator, latent_codes, num_samples, synthesis_kwargs
