import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import utils
ia.seed(1)


def draw_before_after(bbs, image, bbs_aug, image_aug):
    image_before = bbs.draw_on_image(image, thickness=2)
    image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

    cv2.imshow('ImageBefore', image_before)
    cv2.imshow('ImageWindow', image_after)
    cv2.waitKey()

def main():
    file = utils.read_file()
    image1 = file.iloc[0]

    image = cv2.imread(image1['filename'])

    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=image1['xmin'], y1=image1['ymin'], x2=image1['xmax'], y2=image1['ymax'])
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    # Make our sequence deterministic.
    # We can now apply it to the image and then to the BBs and it will
    # lead to the same augmentations.
    # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
    # exactly same augmentations for every batch!
    seq_det = seq.to_deterministic()

    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the
    # functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates

    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
        )
        new_df = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB')

    # image with BBs before/after augmentation (shown below)



if __name__ == '__main__':
    main()