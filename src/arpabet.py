"""
ARPAbet Phoneme Constants
==========================
Shared phoneme-to-viseme mapping used by multiple training modules.
"""

# ARPAbet phoneme → 10-category viseme mapping
ARPABET_TO_VISEME = {
    # aei — open mouth vowels
    "aa": "aei", "ae": "aei", "ah": "aei", "ay": "aei", "ey": "aei",
    "eh": "aei", "ax": "aei",
    # o — rounded vowels
    "ao": "o", "ow": "o", "oy": "o",
    # ee — front high vowels
    "iy": "ee", "ih": "ee",
    # qw — rounded close vowels / w sound
    "uw": "qw", "uh": "qw", "w": "qw",
    # r — r-colored
    "er": "r", "r": "r",
    # l — lateral
    "l": "l",
    # bmp — bilabial stops/nasal
    "b": "bmp", "p": "bmp", "m": "bmp",
    # fv — labiodental fricatives
    "f": "fv", "v": "fv",
    # th — dental fricatives
    "th": "th", "dh": "th",
    # cdgknstxyz — alveolar/velar stops, fricatives, nasals
    "d": "cdgknstxyz", "t": "cdgknstxyz", "n": "cdgknstxyz",
    "k": "cdgknstxyz", "g": "cdgknstxyz", "ng": "cdgknstxyz",
    "s": "cdgknstxyz", "z": "cdgknstxyz",
    "sh": "cdgknstxyz", "zh": "cdgknstxyz",
    "ch": "cdgknstxyz", "jh": "cdgknstxyz",
    "hh": "cdgknstxyz", "y": "cdgknstxyz",
}
