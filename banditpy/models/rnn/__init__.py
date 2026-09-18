"""RNN/ANN model architectures and fitters for bandit tasks.

Public names are re-exported through 'banditpy.models' (the lazy
'__getattr__' facade there) — import from there, not from these submodules
directly, unless you specifically need to avoid the facade's lazy torch
import.
"""
