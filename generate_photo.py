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
    parser.add_argument('-s', '--noise_seed', type=int, default=11, help='Noise seed for randomness.')
    parser.add_argument('-r', '--resolution', type=int, default=1024, help='Output resolution.')
    return parser.parse_args()


def main():
    args = parse_args()
    generate_sample(args.model_name, args.output_directory, args.latent_space_type, args.num_samples, args.noise_seed,
                    args.resolution)


if __name__ == '__main__':
    main()