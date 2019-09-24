"""Audio file(wav) feature extractor."""

import concurrent
import glob
import os

from absl import app
from absl import flags
from absl import logging
import pandas
import tqdm

import feature


flags.DEFINE_string('wav_dir', '', 'Directory to audio files.')
flags.DEFINE_string('csv_output', '', 'Path to csv output.')

FLAGS = flags.FLAGS

def extract_features(filename):
    delay_p, mean, error = feature.extract_features(filename)
    return [filename, delay_p, mean, error]


def main(unused):
    del unused

    wav_files = glob.glob(os.path.join(FLAGS.wav_dir) + '*.wav')

    data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = executor.map(extract_features, wav_files)
        results = list(tqdm.tqdm(futures, total=len(wav_files)))

    for result in results:
        data.append(result)

    dframe = pandas.DataFrame(data,
                              columns=['Name', 'delay_p', 'mean', 'error'])
    dframe.to_csv(FLAGS.csv_output)


if __name__ == '__main__':
    app.run(main)
