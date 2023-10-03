from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
INSTALL_REQUIRES = [
    'jax>=0.4.4',
    'neural_tangents>=0.6.1',
]

setup(
    name='onset-of-variance',
    license='MIT License',
    author='Pehlevan Group',
    author_email='sasha.atan@gmail.com',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/Pehlevan-Group/onset-of-variance',
    long_description=long_description,
    packages=find_packages(),
    long_description_content_type='text/markdown',
    description='Finite Width Effects in Learning Curves of Neural Networks and Kernels',
    python_requires='>=3.6')