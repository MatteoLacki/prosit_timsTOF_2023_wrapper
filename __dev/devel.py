"""
%load_ext autoreload
%autoreload 2
"""

import numpy as np
import tqdm

from prosit_timsTOF_2023_wrapper.main import Prosit2023TimsTofWrapper

sequences = np.array(["PEPTIDE", "PEPTIDECPEPTIDE"])
amino_acid_cnts = np.array(list(map(len, sequences)))
collision_energies = np.array([30.0, 31.1])
charges = np.array([1, 2])

prosit = Prosit2023TimsTofWrapper()
with tqdm.tqdm(total=len(sequences)) as pbar:
    fragment_intensity_lst = list(
        prosit.iter_predict_intensities(
            sequences=sequences,
            amino_acid_cnts=amino_acid_cnts,
            charges=charges,
            collision_energies=collision_energies,
            pbar=pbar,
        )
    )

prosit.get_fragment_intensity_annotations(10, 2, as_ASCI=False)


import re

sequences = np.array(
    [
        "[UNIMOD:121]PEPT[UNIMOD:121]ID[UNIMOD:121]E",
        "PEPTIDECPEPTI[UNIMOD:121]D[UNIMOD:121]E",
    ]
)

cleaned_text = re.sub(r"\[UNIMOD:\d+\]", "", text)


sequences
