import argparse
from utility_functions import generate_sample


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Generate photo of persons that do not exist.')
    parser.add_argument('-m', '--model_name', type=str, default='stylegan_ffhq',
                        choices=['pggan_celebahq','stylegan_celebahq', 'stylegan_ffhq'], help='GAN type.')
    parser.add_argument('-o', '--output_directory', type=str, default='generated_images', help='Output directory.')
    parser.add_argument('-t', '--latent_space_type', type=str, default='W', help='Type of latent vector (Z or W).')
    parser.add_argument('-n', '--num_samples', type=int, default=1, help='Number of photo samples')
    parser.add_argument('-s', '--noise_seed', type=int, default=111, help='Noise seed for randomness.')
    return parser.parse_args()


def main():
    args = parse_args()
    generate_sample(args.model_name, args.output_directory, args.latent_space_type, args.num_samples, args.noise_seed)


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