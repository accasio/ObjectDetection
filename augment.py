import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import utils
import pandas as pd
ia.seed(1)


def draw_before_after(bbs, image, bbs_aug, image_aug):
    image_before = bbs.draw_on_image(image, thickness=2)
    image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

    cv2.imshow('ImageBefore', image_before)
    cv2.imshow('ImageWindow', image_after)
    cv2.waitKey()

def main():
    file = utils.read_file()
    file2 = file.copy()
    image1 = file.iloc[0]

    image = cv2.imread(image1['filename'])

    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=image1['xmin'], y1=image1['ymin'], x2=image1['xmax'], y2=image1['ymax'])
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        )
    ])

    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
        )
        new_name = image1['filename'] + 'generated'

        df = {
            'filename': new_name,
            'xmin': after.x1,
            'ymin': after.y1,
            'xmax': after.x2,
            'ymax': after.y2,
        }

        file2.append(pd.DataFrame(data=df, index=[0]))

    # file2.to_csv(utils.dir + 'test.txt', sep=',', index=False)
    draw_before_after()

if __name__ == '__main__':
    main()