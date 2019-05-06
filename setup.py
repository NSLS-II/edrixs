from os import path
from setuptools import find_packages
import sys
import versioneer
# It needs f2py, https://www.numpy.org/devdocs/f2py/distutils.html
from numpy.distutils.core import Extension
from numpy.distutils.core import setup

# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 6)
if sys.version_info < min_version:
    error = """
edrixs does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*sys.version_info[:2], *min_version)
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

# Python interface to call fortran subroutines
ext_fortran = Extension(name='edrixs.fedrixs',
                        sources=['src/pyapi.f90'])

setup(
    name='edrixs',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="An open source toolkit for simulating RIXS spectra based on ED",
    long_description=readme,
    author="Brookhaven National Lab",
    author_email='yilinwang@bnl.gov',
    url='https://github.com/NSLS-II/edrixs',
    packages=find_packages(exclude=['docs', 'tests', 'bin', 'examples', 'src']),
    entry_points={
        'console_scripts': [
            # 'some.module:some_function',
            ],
        },
    include_package_data=True,
    package_data={
        'edrixs': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
            "edrixs/*",
            "edrixs/atom_data/*",
            ]
        },
    install_requires=requirements,
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    ext_modules=[ext_fortran]
)
