import gzip
import os
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np
from tqdm.autonotebook import tqdm

sumbissions_path = Path('submissions/archive/test/majority_vote')  # or 'submission/archive/heldout/majority_vote'
majority_vote_path = sumbissions_path / 'unzipped'
majority_vote_submission_path = sumbissions_path / 'majority_vote_submission'


def unzip_submsission():
    for sub in tqdm(list(sumbissions_path.glob('*.zip')), desc='unziping submsissions'):
        unzip_path = majority_vote_path / sub.stem
        if unzip_path.exists():
            continue
        with zipfile.ZipFile(sub, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


def unzip_regions():
    for file_path in tqdm(list(majority_vote_path.glob('**/*.gz')), desc='unzipping regions'):
        decompressed_path = file_path.with_suffix('').with_suffix('.h5')
        if decompressed_path.exists():
            continue
        with gzip.open(file_path, 'rb') as f_in, open(decompressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def create_majority_vote():
    regions = set(x.name for x in majority_vote_path.glob('**/*.h5'))
    for region in tqdm(regions, desc='creating majority vote for regions'):
        for year in (2019, 2020):
            predictions = []
            for file_path in majority_vote_path.glob(f'**/{year}/{region}'):
                with h5py.File(file_path, 'r') as f:
                    predictions.append(f['submission'][:])
            predictions = np.stack(predictions, axis=0)
            majority_vote = np.floor(np.mean(predictions, axis=0) + 0.5)  # 0.5 -> 1
            # majority_vote = np.mean(predictions, axis=0).round() # 0.5 -> 0
            assert majority_vote.shape == (60, 32, 252, 252), majority_vote.shape
            assert np.unique(majority_vote).tolist() == [0., 1.], np.unique(majority_vote)
            (majority_vote_submission_path / str(year)).mkdir(parents=True, exist_ok=True)
            with h5py.File(majority_vote_submission_path / str(year) / region, 'w') as f:
                f.create_dataset('submission', data=majority_vote)


def compress_regions():
    for file_path in tqdm(list(majority_vote_submission_path.glob('**/*.h5')), desc='compressing regions'):
        os.system(f'gzip -9f {file_path}')


def compress_submission():
    os.system(f'cd {majority_vote_submission_path}; zip -r ../majority_vote.zip .')


def main():
    majority_vote_path.mkdir(parents=True, exist_ok=True)
    unzip_submsission()
    unzip_regions()
    create_majority_vote()
    compress_regions()
    compress_submission()


if __name__ == '__main__':
    main()
