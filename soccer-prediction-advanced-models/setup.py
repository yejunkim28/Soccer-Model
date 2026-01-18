from setuptools import setup, find_packages

setup(
    name='soccer-prediction-advanced-models',
    version='0.1.0',
    author='yejunkim28',
    author_email='youremail@example.com',
    description='A project to predict soccer player performance using advanced machine learning models.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'tensorflow',  # For neural network model
        'matplotlib',
        'seaborn',
        'pyyaml',
        'jupyter'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)