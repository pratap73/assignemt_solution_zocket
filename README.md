
# Zocket Computer Vision Assignment

first task to generate image using text for advertisment

### Stable Diffusion Text to Image Generation

This script utilizes the Stable Diffusion model to generate images from text prompts.

### Installation

```python


!pip install diffusers --upgrade
!pip install invisible_watermark transformers accelerate safetensors

run each cell of the .ipynb Notebook


To run this script, you'll need to install the following dependencies:

- `diffusers`
- `invisible_watermark`
- `transformers`
- `accelerate`
- `safetensors`
- `Python 3.x`
- `roboflow library`
- `supervision library`
- `opencv-python library`


You can install these dependencies using pip:
```
### Usage

1. Make sure you have a GPU available to run this script efficiently.
2. Run the script in a Python environment, such as Jupyter Notebook or Google Colab.


3. to run `filter-images.py` , Replace `/path/to/input_images`, `/path/to/product_output`, and `/path/to/no_product_output` with the actual paths to your input images directory, product output directory, and no product output directory, respectively.

4. run `gradio_mask_image.py` to mask or segment obejct in a given image



