from setuptools import setup, find_packages

VERSION = '0.0.3'
NAME = 'alm'
IS_RELEASED = False

with open('README.md') as f:
    readme = f.read()

setup(
    name=NAME,
    version=VERSION,
    description='solve analogy with language model',
    long_description=readme,
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    packages=find_packages(exclude=('helper', 'dataset')),
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "transformers",
        "torch",
        "tqdm"
    ]
)
