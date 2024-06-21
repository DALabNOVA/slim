from setuptools import setup, find_packages

setup(
    name='gsgp_slim',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch'
    ],
    entry_points={
        'console_scripts': [
            # If you have any command-line scripts
            # 'command-name = mylibrary.module:function',
        ],
    },
    author= 'Davide Farinati, Diogo Rasteiro, Liah Rosenfeld, Gloria Pietropolli', 
    author_email='dfarinati@novaims.unl.pt, drasteiro@novaims.unl.pt, lrosenfeld@novaims.unl.pt, gloria.pietropolli@phd.units.it',
    description='Semantic Learning algorithm based on Inflate and deflate Mutation (SLIM GSGP)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DALabNOVA/slim',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
