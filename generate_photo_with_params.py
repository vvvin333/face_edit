import argparse
import json
from utility_functions import generate_sample, manipulate_with_params


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Generate photo of persons that do not exist.')
    parser.add_argument('-m', '--model_name', type=str, default='stylegan_ffhq',
                        choices=['pggan_celebahq', 'stylegan_celebahq', 'stylegan_ffhq'], help='GAN type.')
    parser.add_argument('-o', '--output_directory', type=str, default='generated_images', help='Output directory.')
    parser.add_argument('-l', '--latent_file', type=str, default='latents/sample.npy', help='Input latent npy file.')
    parser.add_argument('-p', '--params_file', type=str, default='params.json', help='Input json file with params.')
    parser.add_argument('-t', '--latent_space_type', type=str, default='W', help='Type of latent vector (Z or W).')
    parser.add_argument('-r', '--resolution', type=int, default=1024, help='Output resolution.')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.params_file) as f:
        directions = json.load(f)
    manipulate_with_params(args.latent_file, directions, args.model_name, args.output_directory, args.latent_space_type,
                           args.resolution)

    # # for linear interpolation along directons
    # for attr_name, attr_value in directions.items():
    #     if attr_value == 0:
    #         continue
    #     left_boundary = - abs(attr_value)
    #     delta = abs(attr_value) * 2
    #     for step in range(num_steps):
    #         coeff = left_boundary + step * delta / num_steps
    #         new_images = interpolate(latent_codes, boundaries[attr_name], coeff, generator, synthesis_kwargs)
    #         file_name = os.path.join(output_directory, attr_name + '_' + str(step) + '.jpeg')
    #         new_images = image_processing(new_images, num_samples)
    #         cv2.imwrite(new_images, file_name)
    #         print(file_name, 'saved')
    #


if __name__ == '__main__':
    main()
