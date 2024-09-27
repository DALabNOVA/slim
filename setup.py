from setuptools import setup, find_packages

setup(
    name='gsgp_slim',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            # If you have any command-line scripts
            # 'command-name = mylibrary.module:function',
        ],
    },
    author= 'Liah Rosenfeld, Davide Farinati, Diogo Rasteiro, Gloria Pietropolli',
    author_email='lrosenfeld@novaims.unl.pt, dfarinati@novaims.unl.pt, drasteiro@novaims.unl.pt, gloria.pietropolli@phd.units.it',
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
