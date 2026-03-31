from setuptools import setup, find_packages

# TODO: combine requirements in  the setup file 
setup(
    name='ldm',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)