from setuptools import setup, find_packages

setup(
    name='slim',
    version='0.1',
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
    author='Gloria Pietropolli',
    author_email='gloria.pietropolli@gmail.com',
    description='Semantic Learning algorithm based on Inflate and deflate Mutation (SLIM GSGP)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Rizzolioli/SlimShady',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)