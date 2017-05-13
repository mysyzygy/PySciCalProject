import argparse
from spectral_loudness.audio.engine import Engine


def main():
    """
    spectral_loudness.main() is the main command line entry point to use the spectral_loudness package.
    """
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze spectral loudness of a 16-bit wav file.')
    parser.add_argument('--input_file', '-i', help='Path to input file')
    args = parser.parse_args()

    # create engine object
    engine = Engine(args.input_file)

    # run spectral loudness engine
    engine.run()

if __name__ == "__main__":
    main()
