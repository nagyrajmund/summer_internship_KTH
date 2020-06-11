# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name="aamas20_visualizer",
    version="0.0.1",
    packages=["aamas20_visualizer", "pymo"],
    install_requires=[
        "matplotlib",
        "scipy",
        "pyquaternion",
        "pandas",
        "sklearn",
        "transforms3d",
        "bvh",
    ],
    package_data={"aamas20_visualizer": ["data/data_pipe.sav"]},
    package_dir={"aamas20_visualizer": "aamas20_visualizer"},
)
