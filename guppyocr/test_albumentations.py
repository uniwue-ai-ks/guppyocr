import albumentations as A
from trdg.generators import GeneratorFromStrings


import cv2

augmentations = []

augment = [A.Compose([]),
            #A.Blur(blur_limit=3),
            #A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
            #//A.ISONoise(),
            #A.RandomGravel(),
            #A.
            A.OneOf([
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(brightness_limit=(-0.3,0.7), contrast_limit=(-0.8,0.4),p=1),
                A.RandomFog(fog_coef_lower=0.3,fog_coef_upper=0.4,p=1),    
                A.MultiplicativeNoise(elementwise=True,multiplier=(0.4,4), p=1),       
                A.GaussNoise(var_limit=(10,200),per_channel=False,p=1),
            ]),
            A.OneOf([
            #A.ToGray(),
            ]),
            #A.GaussNoise(),
            A.OneOf([
                A.PixelDropout(dropout_prob=0.3,drop_value=255, p=1),
                A.CoarseDropout(max_holes=20,fill_value=255,p=1),
            ]),
            A.OneOf([
                A.Downscale(0.5,0.5,{"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_LINEAR},p=1),
                A.GridDistortion(interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, normalized=True,p=1)
            ])
]

gen = GeneratorFromStrings(
            [":) Hello World 12 345 !*~"],
            blur=0,
            size=48,
            skewing_angle=1,
            random_skew=False,
            fonts=["/home/norbert/.local/share/fonts/MonoLisa-Regular.ttf"],
            margins=(3,3,3,3)
        )


TEST_EACH=False
if TEST_EACH:

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

    fig = plt.figure(figsize=(5., 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(len(augment), 1),  # creates 2x2 grid of axes
                    axes_pad=0.3,  # pad between axes in inch.
                    )

    for ax, im, name in zip(grid, images, strings):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(name)

    plt.show()
else:
    for a in augment:
        a.p = 0.5
    all_augment = A.Compose(augment)  
    import numpy as np
    img,_ = next(gen)
    img = np.array(img)


    images = [all_augment(image=img)["image"] for _ in range(20)]

    import matplotlib.pyplot as plt
    import numpy as np

    from mpl_toolkits.axes_grid1 import ImageGrid

    im1 = np.arange(100).reshape((10, 10))
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)

    fig = plt.figure(figsize=(5., 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(5,3),  # creates 2x2 grid of axes
                    axes_pad=0.3,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()