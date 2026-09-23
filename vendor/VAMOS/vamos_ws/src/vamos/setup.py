from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup

# Generate setup arguments using catkin's tools
setup_args = generate_distutils_setup(
    packages=find_packages('src'),
    package_dir={'': 'src'},
    scripts=[
        'nodes/interactive_input_pub.py',
    ]
)

# Use setuptools for setup
setup(
    **setup_args,
)