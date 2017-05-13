import argparse
from spectral_loudness.audio.engine import Engine


def main():
    """
    spectral_loudness.main() is the main command line entry point to use the spectral_loudness package.
    """
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze spectral loudness of a 48 kHz, 16-bit stereo wav file.')
    parser.add_argument('--input_file', '-i', help='Path to input file. Must be .wav')
    parser.add_argument('--output_file', '-o', help='Path to output file. Must be .txt or .npy')
    args = parser.parse_args()

    # create engine object
    engine = Engine(input_file=args.input_file, output_file=args.output_file)

    # run spectral loudness engine
    engine.run()

if __name__ == "__main__":
    main()
