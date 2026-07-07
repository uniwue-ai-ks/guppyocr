# Guppy OCR (wip)




## Installation

```
pip install "guppyocr @ git+git@github.com:uniwue-ai-ks/guppyocr.git"
```
or add

```
guppyocr @ guppyocr @ git+git@github.com:uniwue-ai-ks/guppyocr.git
```

to the requirements.txt file


## Usage (Python-API)

```python
import PIL.Image
import torch # it might be important to import torch before some other libraries, such as numpy
import numpy as np
from guppyocr.api import GuppyOCR


model = GuppyOCR.load_model("model.pth") # optional parameter device, e.g. toch.device("cuda") for cuda or torch.device("cpu") for CPU-only
image = np.array(PIL.Image.open("test.png"))

text = model.ocr(image)

```
