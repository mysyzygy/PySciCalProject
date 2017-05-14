import os
import pytest
from ..audio.engine import Engine

INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_files')


@pytest.mark.parametrize('n_filter', [16, 31])
@pytest.mark.parametrize('numtaps', [480])
@pytest.mark.parametrize('buffer_size', [4800])
@pytest.mark.parametrize('wav', ['test_file1.wav'])
def test_engine_run(n_filter, numtaps, buffer_size, wav):
    input_file = os.path.join(INPUT_DIR, wav)
    output_file = os.path.join(INPUT_DIR, wav)[:-4] + '.test'
    engine = Engine(input_file=input_file, buffer_size=buffer_size,
                    n_filter=n_filter, numtaps=numtaps, output_file=output_file)
    engine.run()



