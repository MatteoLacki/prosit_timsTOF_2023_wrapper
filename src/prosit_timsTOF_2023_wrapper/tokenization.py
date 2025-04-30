import re

ALPHABET_UNMOD = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "[]-": 21,
    "-[]": 22,
    "[UNIMOD:737]-": 21,
}


def tokenize_unimod_sequence(
    unimod_sequence: str,
    _token_pattern=re.compile(r"[A-Z](?:\[UNIMOD:\d+\])?"),
) -> list[str]:
    """
    Tokenizes a sequence of modified amino acids.

    Args:
        unimod_sequence: A string representing the sequence of amino acids with modifications.
        _token_pattern: how to fish out an amino acid with its modification.

    Returns:
        A list of tokenized amino acids.
    """
    starts_with_UNIMOD_1 = unimod_sequence.startswith("[UNIMOD:1]")
    return [
        "<START>[UNIMOD:1]" if starts_with_UNIMOD_1 else "<START>",
        *re.findall(
            _token_pattern,
            unimod_sequence[len("[UNIMOD:1]") :]
            if starts_with_UNIMOD_1
            else unimod_sequence,
        ),
        "<END>",
    ]
