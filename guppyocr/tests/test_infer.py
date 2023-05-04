def test_ocr_cuda():
    import PIL.Image
    import torch # it might be important to import torch before some other libraries, such as numpy
    import numpy as np
    from guppyocr.api import GuppyOCR
    model = GuppyOCR.load_model("/home/norbert/uni/page-segment/guppy-models/newseye/model_best.pth") # optional parameter device, e.g. toch.device("cuda") for cuda or torch.device("cpu") for CPU-only
    image = np.array(PIL.Image.open("/home/norbert/uni/page-segment/ocr_datasets/NewsEye/test/ONB_aze_19110701_010_0000015.png").convert("RGB"))

    text = model.ocr(image)
    assert text == "Beitragsbefreiung während eines Streiks die Karenzfrist"


def test_ocr_cpu():
    import PIL.Image
    import torch # it might be important to import torch before some other libraries, such as numpy
    import numpy as np
    from guppyocr.api import GuppyOCR
    model = GuppyOCR.load_model("/home/norbert/uni/page-segment/guppy-models/newseye/model_best.pth") # optional parameter device, e.g. toch.device("cuda") for cuda or torch.device("cpu") for CPU-only
    image = np.array(PIL.Image.open("/home/norbert/uni/page-segment/ocr_datasets/NewsEye/test/ONB_aze_19110701_010_0000015.png").convert("RGB"))

    text = model.ocr(image)
    assert text == "Beitragsbefreiung während eines Streiks die Karenzfrist"