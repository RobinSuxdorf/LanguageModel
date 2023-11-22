from enum import StrEnum

class SpecialTokens(StrEnum):
    '''
    Enum for defining special tokens.
    '''
    PAD = '<PAD>'
    SOS = '<SOS>'
    EOS = '<EOS>'
