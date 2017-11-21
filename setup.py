from distutils.core import setup

setup(name='dimreducer',
      version='1.0',
      description='Dimension reduction methods',
      py_modules=['dimreducer'],
     )

setup(name='multiphenotype_utils',
      version='1.0',
      description='Utility functions for all methods',
      py_modules=['multiphenotype_utils'],
     )

setup(name='general_autoencoder',
      version='1.0',
      description='Autoencoder base class',
      py_modules=['general_autoencoder'],
     )

setup(name='standard_autoencoder',
      version='1.0',
      description='Standard autoencoder',
      py_modules=['standard_autoencoder'],
     )

setup(name='variational_autoencoder',
      version='1.0',
      description='VAE',
      py_modules=['variational_autoencoder'],
     )

setup(name='variational_age_autoencoder',
      version='1.0',
      description='VAE with age',
      py_modules=['variational_age_autoencoder'],
     )
