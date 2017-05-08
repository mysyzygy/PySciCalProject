import argparse
from spectral_loudness import ai
from spectral_loudness import audio
from spectral_loudness import plotting


class Engine():
    def __init__(self, input_file, buffer_size=4800, n_filter=31):
        self.input_file = input_file
        self.buffer_size = buffer_size
        self.n_filter = n_filter

    def run(self):
        print('Running analyzer on input file: {}'.format(self.input_file))


def main():
    parser = argparse.ArgumentParser(description='Analyze spectral loudness of a 16-bit wav file.')
    parser.add_argument('--input_file', '-i', help='Path to input file')
    args = parser.parse_args()

    engine = Engine(args.input_file)
    engine.run()

if __name__ == "__main__":
    main()
