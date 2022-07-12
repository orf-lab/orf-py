from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='orf',
    packages=['orf'],
    version='0.1.2',
    license='MIT',
    description='orf package implements the Ordered Forest estimator for random forest estimation of the ordered choice models.',
    long_description_content_type='text/markdown',
    long_description=long_description,
    author='Michael Lechner, Fabian Muny & Gabriel Okasa',
    author_email='fabian.muny@unisg.ch',
    url='https://orf-lab.github.io',
    #download_url = 'https://github.com/fmuny/ORFpy/archive/refs/tags/v0.1.1.tar.gz',
    keywords=['ordered forest', 'ordered choice', 'random forest'],
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.5',
        'scipy>=1.7.2',
        'scikit-learn>=1.0.2',
        'joblib>=1.0.1',
        'plotnine>=0.8.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
