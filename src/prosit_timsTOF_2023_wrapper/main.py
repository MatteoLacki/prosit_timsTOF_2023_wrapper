import functools
import re
import tqdm
import typing

from collections import namedtuple
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from warnings import warn

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

from cachemir.main import MemoizedOutput
from cachemir.main import get_index_and_stats
from prosit_timsTOF_2023_wrapper.tf_ops import load_model
from prosit_timsTOF_2023_wrapper.tokenization import ALPHABET_UNMOD
from prosit_timsTOF_2023_wrapper.tokenization import tokenize_unimod_sequence


@numba.njit
def get_fragment_intensity_annotations(
    max_ordinal: int,
    max_fragment_charge: int,
    annotations: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Check Prosit2023TimsTofWrapper.get_fragment_intensity_annotations for docs."""
    types, ordinals, charges = annotations[:, :max_ordinal, :, :max_fragment_charge]
    return types.ravel(), ordinals.ravel(), charges.ravel()


@numba.njit
def parse_fragment_intensity_predictions(
    raw_intensities: npt.NDArray,
    max_ordinal: int,
    max_fragment_charge: int,
    max_precursor_sequence_len: int = 30,
    global_fragment_types_cnt: int = 2,
    global_max_fragment_charge: int = 3,
) -> npt.NDArray:
    """
    Retrieve only meaningful values of predictions.

    The predictor returns values that need to be projected onto the physically interpretable space.
    That means that charge of fragments must be <= of the precursor and length of fragment <= that of precursor.

    Arguments;
        raw_intensities (np.array): Raw results from the prosit model.
        max_ordinal (int): Maximal possible ordinal for a given ion.
        max_fragment_charge (int): Maximal possilbe charge for a given ion.

        The rest is coming from Prosit2023TimsTofWrapper.
    """
    raw_intensities = raw_intensities.reshape(
        max_precursor_sequence_len - 1,
        global_fragment_types_cnt,
        global_max_fragment_charge,
    )
    weights = raw_intensities[:max_ordinal, :, :max_fragment_charge].ravel()
    weights[weights < 0] = 0  # this is really fishy that prosit does not return probs.
    return weights


@numba.njit
def iter_parse_fragment_intensity_predictions(
    batch_of_raw_intensities,
    batch_of_max_ordinals,
    batch_of_max_fragment_charges,
    *args,
):
    assert len(batch_of_raw_intensities) == len(batch_of_max_ordinals)
    assert len(batch_of_raw_intensities) == len(batch_of_max_fragment_charges)
    for i in range(len(batch_of_raw_intensities)):
        yield parse_fragment_intensity_predictions(
            batch_of_raw_intensities[i],
            batch_of_max_ordinals[i],
            batch_of_max_fragment_charges[i],
            *args,
        )


@dataclass
class Prosit2023TimsTofWrapper:
    """Wrapper around prosit timsTOF 2023 model.

    This model was downloaded from ZENODO: https://zenodo.org/records/8211811
    PAPER : https://doi.org/10.1101/2023.07.17.549401

    Arguments:
        max_precursor_sequence_len (int): Maximal allowed length of the precursor modelled.
        fragment_types (str): A string with allowed fragment types.
        max_fragment_charge (int): Maximal fragment charge state.
        model_path (str): Path to the serialized model.
        min_collisional_energy_eV (float): Minimal collisional energy used while fitting the model.
        max_collisional_energy_eV (float): Maximal collisional energy used while fitting the model.
        ptm_pattern (str): The regex pattern used to find ptms in sequence strings.
    """

    max_precursor_sequence_len: int = 30
    fragment_types: str = "by"
    max_fragment_charge: int = 3
    model_path: str = files("prosit_timsTOF_2023_wrapper.data").joinpath(
        "Prosit2023TimsTOFPredictor"
    )
    min_collisional_energy_eV: float = 20.81
    max_collisional_energy_eV: float = 69.77
    ptm_pattern: re.Pattern = re.compile("\[UNIMOD:\d+\]")
    cache_path: Path | None = None

    def __post_init__(self):
        self.max_ordinal = self.max_precursor_sequence_len - 1
        self.fragment_cnt = len(self.fragment_types)

    @functools.cached_property
    def annotations(self) -> npt.NDArray:
        """
        Returns:
            np.array: A 4D tensor. First dim: 0 - fragment type, 1 - fragment ordinal, 2 - fragment charge.
            Rest of dims correspond to the output of prosit reshaped into ordinals, types, and charges.
        """
        annotations = np.zeros(
            dtype=np.uint8,
            shape=(
                3,
                self.max_ordinal,
                self.fragment_cnt,
                self.max_fragment_charge,
            ),
        )
        assert "y" in self.fragment_types
        annotations[0, :, 0, :] = ord("y")  # ASCII for y ion
        assert "b" in self.fragment_types
        annotations[0, :, 1, :] = ord("b")  # ASCII for b ion
        for i in range(self.max_ordinal):
            annotations[1, i, :, :] = i + 1  # ordinals
        for i in range(self.max_fragment_charge):
            annotations[2, :, :, i] = i + 1  # charges
        return annotations

    def get_fragment_intensity_annotations(
        self,
        max_ordinal: int,
        max_fragment_charge: int,
        fragment_types_as_uint8: bool = False,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Provide annotations (fragment type, ordinal, and charge) for fragment intensities.

        Arguments:
            max_ordinal (int): The maximal ordinal for a fragment, defined as precursor's amino acid count minus 1.
            max_fragment_charge (int): The maximal fragment charge of the fragment, defined as min(precursor charge, 3).
            fragment_types_as_uint8 (bool): Do not translate fragment types into characters.

        Returns:
            tuple: numpy arrays with fragment type, ordinal, and charge, e.g. b 2 3+.
        """
        assert max_ordinal <= self.max_ordinal
        assert max_fragment_charge <= self.max_fragment_charge
        types, ordinals, charges = get_fragment_intensity_annotations(
            max_ordinal,
            max_fragment_charge,
            self.annotations,
        )
        if not fragment_types_as_uint8:
            types = np.array([chr(x) for x in types], dtype="U1")
        return types, ordinals, charges

    @property
    def model(self):
        return load_model(self.model_path)

    def seq_to_index(
        self,
        seq: str,
        _alphabet: dict[str, int] = ALPHABET_UNMOD,
    ) -> npt.NDArray:
        """Convert a sequence to a list of indices into the alphabet.

        Args:
            seq: A string representing a sequence of amino acids.
            max_length: The maximum length of the sequence to allow.

        Returns:
            A list of integers, each representing an index into the alphabet.
        """
        ret_arr = np.zeros(self.max_precursor_sequence_len, dtype=np.int32)
        tokenized_seq = tokenize_unimod_sequence(seq)[1:-1]
        assert len(tokenized_seq) <= self.max_precursor_sequence_len
        for i, s in enumerate(tokenized_seq):
            ret_arr[i] = _alphabet.get(s, 0)
        return ret_arr

    def iter_predict_intensities(
        self,
        sequences: npt.NDArray,
        charges: npt.NDArray,
        collision_energies: npt.NDArray,
        amino_acid_cnts: npt.NDArray | None = None,
        batch_size: int = 1_024,
        pbar: tqdm.std.tqdm | None = None,
        _ce_normalization: float = 100.0,
        _alphabet: dict[str, int] = ALPHABET_UNMOD,
        **kwargs,
    ) -> typing.Iterator[tuple[npt.NDArray, int, int]]:
        """Iterate over fragment intensity predictions using tensor-flow fitted Prosit timsTOF 2023 model.

        Arguments:
            sequences (npt.NDArray): Array of sequences for which to get the results.
            charges (npt.NDArray): Array of charges sequences come with.
            collision_energies (npt.NDArray): Collision energy per each sequence. We typically use the average of CEs for a precursor ion.
            amino_acid_cnts (npt.NDArray): counts of amino acids per sequence. If None, will be calculated.
            batch_size (int): size of batches for ML calculations,
            pbar (tqdm.std.tqdm|None): TQDM progress bar instance.
            _ce_normalization (float): A normalizing constant for CEs.
            _alphabet (dict): Definition of the alphabet and its mapping to numbers used in tokenization.

        Yields:
            tuple: np.array with Simulated fragment intensities after post-processing and max ordinal and max fragment charge needed for parsing.
        """
        import tensorflow as tf  # on purspose here, do not dare to move it on top of the module!

        assert len(sequences) == len(charges)
        assert len(sequences) == len(collision_energies)

        if amino_acid_cnts is None:
            amino_acid_cnts = np.array(
                [len(re.sub(self.ptm_pattern, "", s)) for s in sequences],
                dtype=np.uint32,
            )
        else:
            assert len(sequences) == len(amino_acid_cnts)

        charges, collision_energies, amino_acid_cnts = (
            xx.to_numpy() if isinstance(xx, pd.Series) else xx
            for xx in (charges, collision_energies, amino_acid_cnts)
        )

        # needed to project the estimated probabilities and parse them
        max_ordinals = np.minimum(amino_acid_cnts - 1, self.max_ordinal)
        max_fragment_charges = np.minimum(charges, self.max_fragment_charge)
        fragment_intensity_cnts = max_ordinals * max_fragment_charges * 2

        if np.any(collision_energies < self.min_collisional_energy_eV):
            cnt = np.sum(collision_energies < self.min_collisional_energy_eV)
            msg = f"Prosit_2023_timsTOF_predictor was trained on CEs above {self.min_collisional_energy_eV} eV. You are out of range in {cnt} cases."
            warn(msg)

        if np.any(collision_energies > self.max_collisional_energy_eV):
            cnt = np.sum(collision_energies > self.max_collisional_energy_eV)
            msg = "Prosit_2023_timsTOF_predictor was trained on CEs below {self.max_collisional_energy_eV} eV. You are out of range in {cnt} cases."
            warn(msg)

        collision_energies_norm = np.expand_dims(
            collision_energies / _ce_normalization, 1
        )

        tf_ds = tf.data.Dataset.from_tensor_slices(
            (
                dict(
                    peptides_in=tf.cast(
                        [self.seq_to_index(s, _alphabet) for s in sequences],
                        dtype=tf.int32,
                    ),
                    precursor_charge_in=tf.one_hot(charges - 1, depth=6),
                    collision_energy_in=tf.cast(
                        collision_energies_norm, dtype=tf.float32
                    ),
                )
            )
        ).batch(batch_size)

        Input = namedtuple("IN", "sequence charge collision_energy")
        Data = namedtuple("DATA", "intensity")
        Stats = namedtuple("STATS", "max_ordinal max_fragment_charge")

        i = 0
        j = 0
        for model_input in tf_ds.map(
            lambda f: (
                f["peptides_in"],
                f["precursor_charge_in"],
                f["collision_energy_in"],
            )
        ):
            next_i = i + batch_size
            for parsed_intensities in iter_parse_fragment_intensity_predictions(
                self.model(list(model_input)).numpy(),  # batch_raw_intensities
                max_ordinals[i:next_i],  # batch_of_max_ordinals
                max_fragment_charges[i:next_i],  # batch_of_max_fragment_charges
                self.max_precursor_sequence_len,
                self.fragment_cnt,
                self.max_fragment_charge,
            ):
                assert len(parsed_intensities) == fragment_intensity_cnts[j]
                yield MemoizedOutput(
                    Input(
                        sequences[j],
                        charges[j],
                        collision_energies[j],
                    ),
                    Stats(max_ordinals[j], max_fragment_charges[j]),
                    pd.DataFrame(dict(intensity=parsed_intensities), copy=False),
                )
                if pbar is not None:
                    pbar.update(1)
                j += 1
            i = next_i

    def get_index_and_stats(
        self,
        sequences: pd.Series | npt.NDArray,
        charges: pd.Series | npt.NDArray,
        collision_energies: pd.Series | npt.NDArray,
        cache_path: Path | str | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        inputs_df = pd.DataFrame(
            dict(
                sequences=sequences,
                charges=charges,
                collision_energies=collision_energies,
            ),
            copy=False,
        )
        if cache_path is None:
            assert self.cache_path is not None
            cache_path = self.cache_path
        cache_path = Path(cache_path)

        index_and_stats, raw_data = get_index_and_stats(
            path=cache_path,
            inputs_df=inputs_df,
            results_iter=self.iter_predict_intensities,
            input_types=dict(sequences=str, charges=int, collision_energies=float),
            stats_types=dict(max_ordinal=int, max_fragment_charge=int),
            verbose=verbose,
        )
        return index_and_stats, raw_data
