from setuptools import setup, find_packages

setup(
    name='trajectory_projection',
    version='0.1.0',
    description='Class for projecting depth and pixel coordinates to 3D points',
    author='UW-Robotics',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python'
    ],
    python_requires='>=3.6',
)
