import albumentations as A
from trdg.generators import GeneratorFromStrings


import cv2

augmentations = []
augment = [A.Compose([]),
            A.CLAHE(p=1),
            #A.Blur(blur_limit=3),
            #A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
            #//A.ISONoise(),
            #A.RandomGravel(),
            #A.
            #A.ToGray(),
            A.MultiplicativeNoise(multiplier=(0.7,1.3), p=1),
            #A.GaussNoise(),
            A.PixelDropout(dropout_prob=0.3,drop_value=255, p=1),
            A.GridDropout(unit_size_min=4,unit_size_max=8,p=1),
            A.CoarseDropout(fill_value=255,p=1),
            A.Downscale(0.25,0.5,p=1),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, normalized=True,p=1)
]

gen = GeneratorFromStrings(
            [":) Hello World 12 345 😃"],
            blur=0,
            size=48,
            skewing_angle=1,
            random_skew=False,
            fonts=["/home/norbert/.local/share/fonts/MonoLisa-Regular.ttf"],
            margins=(3,3,3,3)
        )
import numpy as np
img,_ = next(gen)
img = np.array(img)

images = [a(image=img)["image"] for a in augment]
strings = [str(a) for a in augment]

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid

im1 = np.arange(100).reshape((10, 10))
im2 = im1.T
im3 = np.flipud(im1)
im4 = np.fliplr(im2)

fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(len(augment), 1),  # creates 2x2 grid of axes
                 axes_pad=0.3,  # pad between axes in inch.
                 )

for ax, im, name in zip(grid, images, strings):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)
    ax.set_title(name)

plt.show()