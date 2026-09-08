# VAMOS 

This is the inference and deployment repo for [VAMOS](https://vamos-vla.github.io/).

## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vamos-vla/vamos/blob/main/vamos_demo.ipynb)

### Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:vamos-vla/vamos.git
   cd vamos
   ```

2. **Set up the environment**:

For a minimal installation, either run:
   ```bash
   conda env create -f environment.yml
   conda activate vamos
   ```

Or directly install the necessary packages with pip or your favorite package manager:
```bash
pip3 install torch torchvision transformers peft matplotlib Pillow
```

If you want to test with ROS (Noetic), run the following instead:
   ```bash
   conda env create -f environment_ros.yml
   conda activate vamos
   ```

We use ROS noetic from RoboStack for better portability.

### Quick Test

Test the VLM model with sample images:

```bash
python vamos_demo.py
```

Or use the Jupyter notebook for interactive testing:

```bash
jupyter notebook vamos_demo.ipynb
```

You can access the same notebook on Colab directly from the url at the top of this README.

## ROS Package

VAMOS includes a ROS package. For detailed usage and setup instructions, see [ROS.md](ROS.md).
