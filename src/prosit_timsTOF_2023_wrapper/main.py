import functools
import tqdm
import typing

from dataclasses import dataclass
from importlib.resources import files
from warnings import warn

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd

from prosit_timsTOF_2023_wrapper.tokenization import ALPHABET_UNMOD
from prosit_timsTOF_2023_wrapper.tokenization import tokenize_unimod_sequence


@numba.njit
def get_fragment_intensity_annotations(
    max_ordinal: int,
    max_fragment_charge: int,
    annotations: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    types, ordinals, charges = annotations[:, :max_ordinal, :, :max_fragment_charge]
    return types.ravel(), ordinals.ravel(), charges.ravel()


@numba.njit
def parse_fragment_intensity_predictions(
    raw_intensities: npt.NDArray,
    max_ordinal: int,
    max_fragment_charge: int,
    max_modelled_precursor_sequence_length: int = 30,
    global_fragment_types_cnt: int = 2,
    global_max_fragment_charge: int = 3,
) -> npt.NDArray:
    """
    This is very very non-obvious if we deal correctly with negative values.
    """
    raw_intensities = raw_intensities.reshape(
        max_modelled_precursor_sequence_length - 1,
        global_fragment_types_cnt,
        global_max_fragment_charge,
    )
    weights = raw_intensities[:max_ordinal, :, :max_fragment_charge].ravel()
    weights[weights < 0] = 0
    weights /= weights.max()
    return weights


@numba.njit
def iter_fragment_intensity_predictions(
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


def unpack_dict(features):
    return (
        features["peptides_in"],
        features["precursor_charge_in"],
        features["collision_energy_in"],
    )


@dataclass
class Prosit2023TimsTofWrapper:
    """Wrapper around prosit timsTOF 2023 model.

    This model was downloaded from ZENODO: https://zenodo.org/records/8211811
    PAPER : https://doi.org/10.1101/2023.07.17.549401
    """

    max_precursor_sequence_length: int = 30
    fragment_types: str = "by"
    max_fragment_charge: int = 3
    model_path: str = files("prosit_timsTOF_2023_wrapper.data").joinpath(
        "Prosit2023TimsTOFPredictor"
    )
    min_collisional_energy_eV: float = 20.81
    max_collisional_energy_eV: float = 69.77

    def __post_init__(self):
        self.max_ordinal = self.max_precursor_sequence_length - 1

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
                len(self.fragment_types),
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
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        assert max_ordinal <= self.max_ordinal
        assert max_fragment_charge <= self.max_fragment_charge
        return get_fragment_intensity_annotations(
            max_ordinal,
            max_fragment_charge,
            self.annotations,
        )

    @functools.cached_property
    def model(self):
        import tensorflow as tf  # on purspose here, do not dare to move it on top of the module!

        return tf.saved_model.load(self.model_path)

    def seq_to_index(
        self,
        seq: str,
        _alphabet=ALPHABET_UNMOD,
    ) -> npt.NDArray:
        """Convert a sequence to a list of indices into the alphabet.

        Args:
            seq: A string representing a sequence of amino acids.
            max_length: The maximum length of the sequence to allow.

        Returns:
            A list of integers, each representing an index into the alphabet.
        """
        ret_arr = np.zeros(self.max_precursor_sequence_length, dtype=np.int32)
        tokenized_seq = tokenize_unimod_sequence(seq)[1:-1]
        assert len(tokenized_seq) <= self.max_precursor_sequence_length
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
        _divide_collision_energy_by: float = 100.0,
        _alphabet=ALPHABET_UNMOD,
        **kwargs,
    ) -> typing.Iterator[npt.NDArray]:
        """Iterate over fragment intensity predictions using tensor-flow fitted Prosit timsTOF 2023 model.

        Arguments:
            sequences (npt.NDArray): Array of sequences for which to get the results.
            charges (npt.NDArray): Array of charges sequences come with.
            collision_energies (npt.NDArray): Collision energy per each sequence. We typically use the average of CEs for a precursor ion.
            amino_acid_cnts (npt.NDArray): counts of amino acids per sequence. If None, will be calculated.
            batch_size (int): size of batches for ML calculations,
            pbar (tqdm.std.tqdm|None): TQDM progress bar instance.
            _divide_collision_energy_by (float): A normalizing constant for CEs.

        Yields:
            np.array: Simulated fragment intensities after post-processing.
        """
        import tensorflow as tf  # on purspose here, do not dare to move it on top of the module!

        assert len(sequences) == len(charges)
        assert len(sequences) == len(collision_energies)
        assert len(sequences) == len(amino_acid_cnts)

        charges, collision_energies, amino_acid_cnts = (
            xx.to_numpy() if isinstance(xx, pd.Series) else xx
            for xx in (charges, collision_energies, amino_acid_cnts)
        )

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

        # we could make an sklearn pipeline out of that...
        collision_energies_norm = np.expand_dims(
            collision_energies / _divide_collision_energy_by, 1
        )

        # shitty interfaces = interfeces
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

        i = 0
        j = 0
        for model_input in tf_ds.map(unpack_dict):
            for parsed_intensities in iter_fragment_intensity_predictions(
                self.model(list(model_input)).numpy(),  # batch_raw_intensities
                max_ordinals[i : i + batch_size],  # batch_of_max_ordinals
                max_fragment_charges[
                    i : i + batch_size
                ],  # batch_of_max_fragment_charges
                self.max_precursor_sequence_length,
                len(self.fragment_types),
                self.max_fragment_charge,
            ):
                assert len(parsed_intensities) == fragment_intensity_cnts[j]
                yield parsed_intensities
                if pbar is not None:
                    pbar.update(1)
                j += 1
            i += batch_size
