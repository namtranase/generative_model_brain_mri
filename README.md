# generative_model_brain_mri
Using 3 types of GAN for generate MRI Brain images.
Including: DCGAN, normal_GAN and PGGAN.

## Quick Start

### Setup Project

Clone source code from github:

```bash
git clone https://github.com/namtranase/generative_model_brain_mri.git
cd generative_model_brain_mri
```

Create virtual environment to install dependencies for project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run classification program

Classification program with VGG16 model

```bash
PYTHONPATH=. ./bin/classify_validation
```
