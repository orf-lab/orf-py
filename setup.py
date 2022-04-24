from distutils.core import setup

setup(
  name = 'orf',
  packages = ['orf'],
  version = '0.1.0',
  license = 'MIT',
  description = 'orf package implements the Ordered Forest estimator for random forest estimation of the ordered choice models.',
  author = 'Michael Lechner, Fabian Muny & Gabriel Okasa',
  author_email = 'your.email@domain.com',
  url = 'https://github.com/fmuny/ORFpy',
  download_url = 'https://github.com/fmuny/ORFpy/archive/refs/tags/v0.1.0.tar.gz',
  keywords = ['ordered forest', 'ordered choice', 'random forest'],
  install_requires=[
          'numpy>=1.20.3',
          'pandas>=1.3.2',
          'scipy>=1.7.1',
          'scikit-learn>=0.24.2',
          'econml>=0.12.0',
          'joblib>=1.0.1',
          'plotnine>=0.8.0',
          'multiprocessing>=2.6.2.1',
          'sharedmem>=0.3.8',
          
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
