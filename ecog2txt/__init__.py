# standard libraries
import os

# third-party packages
import pandas as pd

from tensor2tensor.data_generators import text_encoder

# paths
text_dir = os.path.join(os.path.dirname(__file__), 'auxiliary')

# other useful variables
EOS_token = text_encoder.EOS
pad_token = text_encoder.PAD
OOV_token = '<OOV>'

# useful sets
TOKEN_TYPES = {
    'phoneme', 'word', 'trial', 'word_sequence', 'word_piece_sequence',
    'phoneme_sequence'
}
DATA_PARTITIONS = {'training', 'validation', 'testing'}

# useful linguistic things
consonant_dict = {
    'phoneme': [
        'p', 'b', 't', 'd', 'k', 'g',
        'f', 'v', '\u03B8', '\u00F0', 's', 'z', '\u0283', '\u0292', 'h',
        't\u0283', 'd\u0292',
        'm', 'n', '\u014b',
        'l', 'r',  # '\u0279',
        'w', 'j',
    ],
    'voicing': [
        'voiceless', 'voiced', 'voiceless', 'voiced', 'voiceless', 'voiced',
        'voiceless', 'voiced', 'voiceless', 'voiced', 'voiceless',
        'voiced', 'voiceless', 'voiced', 'voiceless',
        'voiceless', 'voiced',
        'voiced', 'voiced', 'voiced',
        'voiced', 'voiced',
        'voiced', 'voiced',
    ],
    'place': [
        'bilabial', 'bilabial', 'alveolar', 'alveolar', 'velar', 'velar',
        'labiodental', 'labiodental', 'dental', 'dental', 'alveolar',
        'alveolar', 'palatal', 'palatal', 'glotal',
        'palatal', 'palatal',
        'bilabial', 'alveolar', 'velar',
        'alveolar', 'palatal',
        'labio-velar', 'palatal'
    ],
    'manner': [
        'stop', 'stop', 'stop', 'stop', 'stop', 'stop',
        'fricative', 'fricative', 'fricative', 'fricative', 'fricative',
        'fricative', 'fricative', 'fricative', 'fricative',
        'affricate', 'affricate',
        'nasal', 'nasal', 'nasal',
        'liquid', 'liquid',
        'approximant', 'approximant',
    ],
    'ARPABET': [
        'p', 'b', 't', 'd', 'k', 'g',
        'f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh', 'hh',
        'ch', 'jh',
        'm', 'n', 'ng',
        'l', 'r',
        'w', 'y',
    ]
}
consonant_df = pd.DataFrame(consonant_dict)

# "Acoustic Characteristics of American English Vowels"
# Hillenbrand et al
# J. Acoustic Soc. Am., 97(5), Pt. 1
# 1995
vowel_dict = {
    'phoneme': ['i', '\u026A', 'e', '\u025B', '\u00E6', '\u0251', '\u0252', '\u0254', 'o', '\u028A', 'u', '\u028C'],
    'F1': [342, 427, 476, 580, 588, 768, 768, 652, 497, 469, 378, 623], 
    'F2': [2322, 2034, 2089, 1799, 1952, 1333, 1333, 997, 910, 1122, 997, 1200],
}
# '\u0259'? 'a'?
vowel_df = pd.DataFrame(vowel_dict)
