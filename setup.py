
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='relernntest',
      version='0.1',
      description='ReLERNN',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/kern-lab/ReLERNN/',
      author='Jeff Adrion, Jared Galloway',
      author_email='',
      license='MIT',      
      packages=find_packages(exclude=[]),
      install_requires=[
            "msprime",
            "numpy",
            "h5py",
            "scikit-allel",
            "matplotlib",
            "sklearn",
            "keras"],
      scripts=[
            "ReLERNN_SIMULATE",    
            "ReLERNN_TRAIN",    
            "ReLERNN_PREDICT",  
            "ReLERNN_BSCORRECT"],
      zip_safe=False,
      setup_requires=[],
)

