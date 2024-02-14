from setuptools import setup, find_packages

setup(
    name='moldyn',
    version='1.0.0',
    author='Kevin Roice',
    author_email='roice@ualberta.ca',
    description='A package to predict molecular energies and force fields',
    packages=find_packages(),
    install_requires=[
        # These will be installed whenever this package is pip installed.
       'ase==3.22.1',
        'e3x==1.0.1',
        'py3Dmol==2.0.4',
        'wandb==0.16.3',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
