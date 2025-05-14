"""
%load_ext autoreload
%autoreload 2
"""

import numba
import numpy as np
import tqdm

from prosit_timsTOF_2023_wrapper.main import Prosit2023TimsTofWrapper
from prosit_timsTOF_2023_wrapper.normalizations import normalize_to_max
from prosit_timsTOF_2023_wrapper.normalizations import normalize_to_sum

sequences = np.array(["PEPTIDE", "PEPTIDECPEPTIDE"])
amino_acid_cnts = np.array(list(map(len, sequences)))
collision_energies = np.array([30.0, 31.1])
charges = np.array([1, 2])

prosit = Prosit2023TimsTofWrapper()


with tqdm.tqdm(total=len(sequences)) as pbar:
    results = [
        (
            normalize_to_max(fragment_intensities),
            prosit.get_fragment_intensity_annotations(max_ordinal, max_charge),
        )
        for fragment_intensities, max_ordinal, max_charge in prosit.iter_predict_intensities(
            sequences=sequences,
            # amino_acid_cnts=amino_acid_cnts,
            charges=charges,
            collision_energies=collision_energies,
            pbar=pbar,
        )
    ]

prosit.get_fragment_intensity_annotations(10, 2, fragment_types_as_uint8=True)
prosit.get_fragment_intensity_annotations(10, 2, fragment_types_as_uint8=False)

# OK, now make some cache of the results.


import re

sequences = np.array(
    [
        "[UNIMOD:121]PEPT[UNIMOD:121]ID[UNIMOD:121]E",
        "PEPTIDECPEPTI[UNIMOD:121]D[UNIMOD:121]E",
    ]
)

cleaned_text = [re.sub(r"\[UNIMOD:\d+\]", "", sequence) for sequence in sequences]


sequences
from prosit_timsTOF_2023_wrapper.tokenization import ALPHABET_UNMOD
