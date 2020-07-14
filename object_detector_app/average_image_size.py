import glob
import cv2

IMAGES_PATH = "test_input/kitti_single/training/image_2/*.png"

def main():
    image_paths = sorted(glob.glob(IMAGES_PATH))[0:500]
    images = []
    for image_path in image_paths:
        images.append(cv2.imread(image_path))

    h_sum = sum([image.shape[0] for image in images])
    w_sum = sum([image.shape[1] for image in images])

    print("average height: {}".format(h_sum / len(images)))
    print("average width: {}".format(w_sum / len(images)))

if __name__ == "__main__":
    main()
