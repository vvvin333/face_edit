import argparse
import json
from utility_functions import linear_interpolations


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Generate photo of persons that do not exist.')
    parser.add_argument('-m', '--model_name', type=str, default='stylegan_ffhq',
                        choices=['pggan_celebahq', 'stylegan_celebahq', 'stylegan_ffhq'], help='GAN type.')
    parser.add_argument('-o', '--output_directory', type=str, default='generated_images', help='Output directory.')
    parser.add_argument('-l', '--latent_file', type=str, default='latents/sample_base.npy', help='Input latent npy file.')
    parser.add_argument('-p', '--params_file', type=str, default='params.json', help='Input json file with params.')
    parser.add_argument('-n', '--num_steps', type=int, default=6, help='Steps number for linear interpolations.')
    parser.add_argument('-t', '--latent_space_type', type=str, default='W', help='Type of latent vector (Z or W).')
    parser.add_argument('-r', '--resolution', type=int, default=1024, help='Output resolution.')
    parser.add_argument('-s', '--show_interpolations', type=str, default='false', help='If show interpolation table.')
    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def main():
    args = parse_args()
    with open(args.params_file) as f:
        directions = json.load(f)
    linear_interpolations(args.latent_file, directions, args.num_steps, args.model_name, args.output_directory,
                          args.latent_space_type, args.resolution, str2bool(args.show_interpolations))


if __name__ == '__main__':
    main()
