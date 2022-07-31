# Module which contains the functions used for post-processing
import pydensecrf.densecrf as dcrf
import numpy as np
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from PIL import Image
import albumentations as album
from glob import glob

def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return (
        np.stack(
            [np.array(Image.open(f)) for f in sorted(glob(path + "/*.png"))]
        ).astype(np.float32)
        / 255.0
    )

def visualize(index, save, show, **images):
    """
    Plot images in one row
    """
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0.005, hspace=0.005)
    plt.tight_layout()
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx+1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        #plt.title(name.replace("_", ""), fontsize=50)
        #if image.shape[0] != 400:
        #    image = image.T
        plt.imshow(image)
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig('examples/'+str(index)+'.png', bbox_inches='tight')


def vis_models():
    predictions_unet = np.load("unet_predictions_final.npy")
    predictions_maskformer = np.load("maskformer_predictions_final.npy")
    predictions_segformer = np.load("segformer_predictions_final.npy")

    predictions_maskformer = np.reshape(predictions_maskformer, (144, 400,400))

    test_path = 'data/test/images'
    test_images = load_all_from_path(test_path)


    for i in [1, 11, 32]:

        visualize(
            index=i,
            save=True,
            show=False,
            Original =test_images[i],
            UNet=predictions_unet[i],
            SegFormer=predictions_segformer[i],
            MaskFormer=predictions_maskformer[i],
            )

def vis_album():
    pillow_image = Image.open("data/validation/images/satimage_106.png")
    image = np.array(pillow_image)

    transform1 = album.OneOf(
                [
                    album.HorizontalFlip(p=1),
                    album.VerticalFlip(p=1),
                    album.RandomRotate90(p=1),
                ],
                p=1,
    )
    transform2 =album.OneOf(
                [
                    album.MotionBlur(p=1),
                    album.MedianBlur(blur_limit=3, p=1),
                    album.Blur(blur_limit=3, p=1),
                ],
                p=1,
    )
    transform3=album.OneOf(
                [
                    album.OpticalDistortion(p=1),
                    album.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1),
                ],
                p=1,
            )
    transform4 = album.OneOf(
                [
                    album.RandomBrightness(limit=0.2, p=1),
                ],
                p=1,
            )
    transformed_image_1 = transform1(image=image)['image']
    transformed_image_2 = transform2(image=image)['image']
    transformed_image_3 = transform3(image=image)['image']
    transformed_image_4 = transform4(image=image)['image']

    visualize(
        index=1,
        save=True,
        show=True,
        Flip = transformed_image_1,
        Blur = transformed_image_2,
        Distortion = transformed_image_3,
        Brightness = transformed_image_4,
    )
    plt.show()


def vis_challenges():
    image1 = np.array(Image.open("data/test/images/satimage_145.png"))
    image2 = np.array(Image.open("data/test/images/satimage_155.png"))
    image3 = np.array(Image.open("data/test/images/satimage_176.png"))
    a = "Tiny roads"
    b = "Occlusion"
    c = "Colors"
    visualize(
        index=1,
        save=True,
        show=False,
        a = image1,
        b = image2,
        c = image3,
    )

def main():
    vis_models()
    vis_album()
    vis_challenges()

if __name__ == "__main__":
    main()