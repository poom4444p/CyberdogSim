from setuptools import setup, find_packages

setup(
    name='trajectory_prediction',
    version='0.1.0',
    description='Class for handling navigation with 2d trajectory prediction',
    author='UW-Robotics',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python'
    ],
    python_requires='>=3.6',
)
