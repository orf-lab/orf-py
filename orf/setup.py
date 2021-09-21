from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("honest_fit.pyx"),
    include_dirs=[numpy.get_include()]
)