from setuptools import setup, find_packages

setup(
    name="speech_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'uvicorn',
        'joblib',
        'scikit-learn',
        'numpy',
        'requests'
    ],
)