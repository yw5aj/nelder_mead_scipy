# nelder_mead_scipy
Nelder Mead optimization algorithm extracted from SciPy for stand-alone use with NumPy + Python only, where installing SciPy might be difficult (e.g. in ABAQUS).

This code was literally copy-and-pasted from https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py with minimal cleaning. The cleaning was done so that the algorithm runs with only NumPy and Python available.

I wrote this to be used for my research in ABAQUS. Because the finite element model results usually do not have nice gradient informations (and numerical differentiations could be noisy), Nelder Mead is a great fit.

# Licence
Since this is copy-and-pasted from SciPy, it should be under SciPy license. See LICENSE.MD for more details.


