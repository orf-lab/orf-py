from distutils.core import setup

setup(
  name = 'orf',
  packages = ['orf'],
  version = '0.1.0',
  license = 'MIT',
  description = 'orf package implements the Ordered Forest estimator for random forest estimation of the ordered choice models.',
  author = 'Michael Lechner, Fabian Muny & Gabriel Okasa',
  author_email = 'fabian.muny@unisg.ch',
  url = 'https://github.com/fmuny/ORFpy',
  download_url = 'https://github.com/fmuny/ORFpy/archive/refs/tags/v0.1.0.tar.gz',
  keywords = ['ordered forest', 'ordered choice', 'random forest'],
  install_requires=[
          'numpy>=1.20.3',
          'pandas>=1.4.2',
          'scipy>=1.8.0',
          'scikit-learn>=1.0.2',
          'econml>=0.13.0',
          'joblib>=1.1.0',
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
