import os
import cv2


def load_images_from_folder(folder, eyes=0):
    count = 0
    error_count = 0
    images = []
    for filename in os.listdir(folder):
        try:
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.resize(img, (80, 80))  ## Resizing the images
            ## for eyes if it is 0: open, 1: close
            images.append([img, eyes])
        except:
            error_count += 1
            print('ErrorCount = ' + str(error_count))
            continue

        count += 1
        if count % 500 == 0:
            print('Succesful Image Import Count = ' + str(count))

    return images
