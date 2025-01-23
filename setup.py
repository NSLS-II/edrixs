import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# needed for setuptools.build_meta to pickup vendored versioneer.py
sys.path.insert(0, os.path.dirname(__file__))
import versioneer  # noqa: E402

# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 10)
if sys.version_info < min_version:
    error = """
edrixs does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(
        *sys.version_info[:2], *min_version
    )
    sys.exit(error)

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open(os.path.join(here, "requirements.txt")) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line
        for line in requirements_file.read().splitlines()
        if not line.startswith("#")
    ]


# adapted from https://martinopilia.com/posts/2018/09/15/building-python-extension.html
# see also https://github.com/pyscf/pyscf/blob/master/setup.py
class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir=".", sources=[], **kwargs):
        Extension.__init__(self, name, sources=sources, **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            _ = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable")

        for ext in self.extensions:

            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

            cmake_args = [
                "-DEDRIXS_PY_INTERFACE=ON",
                # Ask CMake to place the resulting library in the directory containing the extension
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
                # static libraries are placed in a temporary build directory
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}".format(self.build_temp),
                # Don't need executables for python lib
                "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}".format(self.build_temp),
            ]

            configure_args = os.getenv("CMAKE_CONFIGURE_ARGS")
            if configure_args:
                cmake_args.extend(configure_args.split(" "))

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config
            subprocess.check_call(
                ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp
            )

            # Build
            subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)


setup(
    name="edrixs",
    version=versioneer.get_version(),
    description="An open source toolkit for simulating RIXS spectra based on ED",
    long_description=readme,
    author="Brookhaven National Lab",
    author_email="yilinwang@bnl.gov",
    url="https://github.com/NSLS-II/edrixs",
    packages=find_packages(exclude=["docs", "tests", "bin", "examples", "src"]),
    entry_points={
        "console_scripts": [
            "ed.x=edrixs.scripts:ed",
            "xas.x=edrixs.scripts:xas",
            "rixs.x=edrixs.scripts:rixs",
            "opavg.x=edrixs.scripts:opavg",
        ],
    },
    include_package_data=True,
    package_data={
        "edrixs": [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
            "edrixs/*",
            "edrixs/atom_data/*",
        ]
    },
    install_requires=requirements,
    license="GPL-3.0-or-later",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    ext_modules=[
        CMakeExtension("edrixs.placeholder")
    ],  # edrixs.foo puts build outputs under edrixs subdir
    cmdclass={"build_ext": cmake_build_ext},
)
