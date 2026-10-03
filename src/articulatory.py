"""
Articulatory Feature Encoding
==============================
Encodes ARPAbet phonemes as articulatory feature vectors instead of one-hot.

Each phoneme maps to a short feature vector describing how it's physically
produced: where in the mouth (place), how air flows (manner), whether the
vocal cords vibrate (voicing), and mouth shape (rounding, height, backness).

This lets the model learn that "b" and "p" share lip closure, and that
rare phonemes can borrow from articulatory neighbors.

Feature dimensions:
  Consonants: [place(8), manner(6), voicing(1)] = 15 dims
  Vowels:     [height(3), backness(3), rounding(1)] = 7 dims
  Combined:   15 + 7 = 22 dims (unused slots zero-filled)

Compared to one-hot (~40 dims per slot, ~120 total for prev+curr+next),
articulatory features are 66 dims total and encode similarity.
"""

import numpy as np

# -- Consonant features --
# Place of articulation (one-hot, 8 categories)
_PLACE = {
    "bilabial": 0,
    "labiodental": 1,
    "dental": 2,
    "alveolar": 3,
    "postalveolar": 4,
    "velar": 5,
    "glottal": 6,
    "palatal": 7,
}

# Manner of articulation (one-hot, 6 categories)
_MANNER = {
    "stop": 0,
    "fricative": 1,
    "affricate": 2,
    "nasal": 3,
    "approximant": 4,
    "lateral": 5,
}

# -- Vowel features --
# Height (one-hot, 3 levels)
_HEIGHT = {"high": 0, "mid": 1, "low": 2}

# Backness (one-hot, 3 levels)
_BACKNESS = {"front": 0, "central": 1, "back": 2}

N_PLACE = len(_PLACE)
N_MANNER = len(_MANNER)
N_HEIGHT = len(_HEIGHT)
N_BACKNESS = len(_BACKNESS)

# Total feature vector: place(8) + manner(6) + voicing(1) + height(3) + backness(3) + rounding(1)
FEATURE_DIM = N_PLACE + N_MANNER + 1 + N_HEIGHT + N_BACKNESS + 1  # = 22

# Consonant definitions: (place, manner, voiced)
_CONSONANTS = {
    "b":  ("bilabial",     "stop",        True),
    "p":  ("bilabial",     "stop",        False),
    "m":  ("bilabial",     "nasal",       True),
    "f":  ("labiodental",  "fricative",   False),
    "v":  ("labiodental",  "fricative",   True),
    "th": ("dental",       "fricative",   False),
    "dh": ("dental",       "fricative",   True),
    "t":  ("alveolar",     "stop",        False),
    "d":  ("alveolar",     "stop",        True),
    "n":  ("alveolar",     "nasal",       True),
    "s":  ("alveolar",     "fricative",   False),
    "z":  ("alveolar",     "fricative",   True),
    "l":  ("alveolar",     "lateral",     True),
    "r":  ("alveolar",     "approximant", True),
    "sh": ("postalveolar", "fricative",   False),
    "zh": ("postalveolar", "fricative",   True),
    "ch": ("postalveolar", "affricate",   False),
    "jh": ("postalveolar", "affricate",   True),
    "k":  ("velar",        "stop",        False),
    "g":  ("velar",        "stop",        True),
    "ng": ("velar",        "nasal",       True),
    "w":  ("bilabial",     "approximant", True),
    "y":  ("palatal",      "approximant", True),
    "hh": ("glottal",      "fricative",   False),
}

# Vowel definitions: (height, backness, rounded)
_VOWELS = {
    "iy": ("high",  "front",   False),
    "ih": ("high",  "front",   False),
    "ey": ("mid",   "front",   False),
    "eh": ("mid",   "front",   False),
    "ae": ("low",   "front",   False),
    "aa": ("low",   "back",    False),
    "ah": ("mid",   "central", False),
    "ax": ("mid",   "central", False),
    "ao": ("mid",   "back",    True),
    "ow": ("mid",   "back",    True),
    "uh": ("high",  "back",    True),
    "uw": ("high",  "back",    True),
    "er": ("mid",   "central", False),
    "oy": ("mid",   "back",    True),
    "ay": ("low",   "central", False),
}

_cache = {}


def encode(phone: str) -> np.ndarray:
    """
    Encode a single ARPAbet phoneme as an articulatory feature vector.

    Returns a float32 array of shape (FEATURE_DIM,). Unknown phonemes
    (including "sil") return a zero vector.
    """
    if phone in _cache:
        return _cache[phone]

    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    offset = 0

    if phone in _CONSONANTS:
        place, manner, voiced = _CONSONANTS[phone]
        vec[offset + _PLACE[place]] = 1.0
        offset += N_PLACE
        vec[offset + _MANNER[manner]] = 1.0
        offset += N_MANNER
        vec[offset] = 1.0 if voiced else 0.0

    elif phone in _VOWELS:
        height, backness, rounded = _VOWELS[phone]
        offset = N_PLACE + N_MANNER + 1  # skip consonant slots
        vec[offset + _HEIGHT[height]] = 1.0
        offset += N_HEIGHT
        vec[offset + _BACKNESS[backness]] = 1.0
        offset += N_BACKNESS
        vec[offset] = 1.0 if rounded else 0.0

    _cache[phone] = vec
    return vec


def encode_triplet(prev: str, curr: str, nxt: str) -> np.ndarray:
    """Encode a phoneme with its context as a concatenated feature vector."""
    return np.concatenate([encode(prev), encode(curr), encode(nxt)])


# Total dims for a (prev, curr, next) triplet + duration + position
TRIPLET_DIM = FEATURE_DIM * 3 + 4  # 22*3 + 4 = 70
