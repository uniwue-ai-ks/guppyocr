# Guppy OCR (wip)




## Installation

```
pip install "guppyocr @ git+https://gitlab+deploy-token-113:pdXYgWy8yHMq5it7CxK6@gitlab2.informatik.uni-wuerzburg.de/nof36hn/guppyocr.git"
```
or add

```
guppyocr @ guppyocr @ git+https://gitlab+deploy-token-113:pdXYgWy8yHMq5it7CxK6@gitlab2.informatik.uni-wuerzburg.de/nof36hn/guppyocr.git
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