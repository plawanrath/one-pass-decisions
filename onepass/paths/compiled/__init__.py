"""Compiled CPU path: the packed encoder as an integer kernel with the weights as a linked constant.

One deployment path among the others (\S Setup). `reference.py` is the NumPy model the generated C matches bit for bit;
`emit.py` writes the C, the header and the weight blob, compiles them, and runs the resulting binary on packed inputs.
"""
from .reference import MBPacked, forward, pack  # noqa: F401
