from setuptools import setup, find_packages


# Requires conda-build version 2.0.9
# Build conda package with following commands in shyft_viz\:
# python setup.py install
# python setup.py bdist_conda


# VERSION should be set in a previous build step (ex: TeamCity)
VERSION = open('VERSION').read().strip()

# Create temporary file
open('version.py', 'w').write('__version__ = "%s"\n' % VERSION)


setup(
    name='shyft_viz',
    version=VERSION,
    author='Statkraft',
    author_email='shyft@statkraft.com',
    url='https://github.com/yisak/shyft_viz',
    description='Matplotlib-based tool for interactive visualization of data in a shyft region-model',
    license='LGPL v3',
    packages=find_packages(),
    package_data={},
    entry_points={},
    requires=[]
)
