#!/usr/bin/env python3

import os
import cv2
import numpy as np


def save_segments(image, rectangles, output_path):
    for i, rect in enumerate(rectangles):
        x, y, w, h = cv2.boundingRect(rect)
        segment = image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_path, f'segment_{i}.tif'), segment)


def is_box_inside(box1, box2):
    x1, y1, w1, h1 = cv2.boundingRect(box1)  # inside
    x2, y2, w2, h2 = cv2.boundingRect(box2)  # outside
    if x1 >= x2-5 and y1 >= y2-5 and x1+w1 <= x2+w2+5 and y1+h1 <= y2+h2+5:
        return True


def get_boxes(img_path):
    #print(img_path)
    if not os.path.exists(img_path):
        return None

    im = cv2.imread(img_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_canny = cv2.Canny(im_gray, 0, 150)
    kernel = np.ones((5, 5), np.uint8)
    im_dilate = cv2.dilate(im_canny, kernel, iterations=5)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        im_dilate,
        connectivity=4,
        ltype=cv2.CV_32S
    )

    total_area = im_dilate.shape[0] * im_dilate.shape[1]
    min_area = total_area // 175

    boxes = []
    for i in range(num_labels):
        x, y, w, h, _ = stats[i]
        box = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])
        boxes.append(box)

    # remove segments that are very narrow
    minimum_ratio = 0.15
    ratios = []
    for i in range(num_labels):
        x, y, w, h, _ = stats[i]
        ratios.append(
            min(w, h) / (w + h)
        )
    boxes = [
        box for i, box in enumerate(boxes)
        if ratios[i] >= minimum_ratio
    ]

    # remove box identical with image
    boxes = [box for box in boxes if tuple(box[0]) != (0, 0) and tuple(box[2]) != tuple(im.shape)]

    # remove boxes that are too small
    boxes = [box for box in boxes if cv2.contourArea(box) >= min_area]

    boxes = sorted(boxes, key=cv2.contourArea, reverse=True)

    # drop boxes that are > 70% of image
    if len(boxes) > 1:
        boxes = [box for box in boxes if cv2.contourArea(box) <= total_area * 0.7]

    # russian dolls
    to_drop = []
    hierarchy = []
    for i, outside_box in enumerate(boxes):
        inside = []
        for ii, box2 in enumerate(boxes):
            if ii == i:
                continue
            if is_box_inside(box2, outside_box):
                # only append direct children
                if not any(is_box_inside(box2, boxes[iii]) for iii in inside):
                    inside.append(ii)
        hierarchy.append((i, inside))
    for i, children in hierarchy[::-1]:
        # no inner box or more than one, continue
        if len(children) != 1:
            continue

        ii = children[0]
        # inner box has many children
        #if len(hierarchy[ii]) >

        area_parent = cv2.contourArea(boxes[i])
        area_child = cv2.contourArea(boxes[ii])
        if area_child / area_parent < 0.1:
            # too small inner box, keep only outer box
            to_drop.append(ii)
        elif area_child / area_parent > 0.45:
            # inner box fills outer box, keep only inner box
            to_drop.append(i)
        else:
            pass

    for ix in sorted(set(to_drop), reverse=True):
        boxes.pop(ix)

    if len(boxes) > 1:
        copy = boxes[:]
        for i, box in enumerate(boxes):
            # if there are more than one other box
            if len(boxes[i+1:]) < 2:
                break
            # and if the current box contains all the other ones…
            if all([is_box_inside(box2, box) for box2 in boxes[i+1:]]):
                # and the inside boxes cover at least 20% of area of current box
                # then drop it
                if sum(cv2.contourArea(box2) for box2 in boxes[i+1:]) / cv2.contourArea(box) >= 0.2:
                    copy = boxes[i+1:]
        boxes = copy

    if len(boxes) == 2:
        c1, c2 = boxes
        if cv2.contourArea(c2) < total_area // 100 and is_box_inside(c2, c1):
            boxes = [c1]

    if len(boxes) == 2:
        c1, c2 = boxes
        if cv2.contourArea(c1) / total_area > 0.5 and is_box_inside(c2, c1):
            boxes = [c2]

    # drop remaining boxes that are > 60% of image
    if len(boxes) > 1:
        boxes = [box for box in boxes if cv2.contourArea(box) <= total_area * 0.6]

    return im, np.array(boxes)


def segment_image(image_path, output_directory):
    image, boxes = get_boxes(image_path)
    if boxes is not None:
        image_directory = os.path.join(
            output_directory,
            os.path.splitext(os.path.basename(image_path))[0]
        )
        os.mkdir(image_directory)
        save_segments(image, boxes, image_directory)
    return True, boxes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_directory', help='directory with images to segment')
    parser.add_argument('output_directory', help='directory where segments will be saved')
    parser.add_argument(
        '--save_boxes',
        default='',
        help='path for additionnally saving coordinates of boxes'
    )
    parser.add_argument('-j', '--jobs', type=int, default=1, help="parallel processing")
    args = parser.parse_args()

    from tqdm import tqdm

    images = [os.path.join(args.image_directory, name)
              for name
              in sorted(os.listdir(args.image_directory))]
    os.makedirs(args.output_directory, exist_ok=True)

    if args.jobs == 1:
        all_boxes = []
        for path in tqdm(images):
            result, boxes = segment_image(path, args.output_directory)
            if args.save_boxes:
                all_boxes.append(boxes)
        if args.save_boxes:
            np.savez(args.save_boxes, **dict(zip(images, all_boxes)))

    elif args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            batch_size = 100
            n_images = len(images)
            n_batches = (n_images // batch_size) + 1

            all_boxes = []
            all_images = []

            with tqdm(total=n_images) as bar:

                for i in range(n_batches):
                    start = i*batch_size
                    end = min(start+batch_size, n_images)

                    future_results = {
                        executor.submit(segment_image, path, args.output_directory): path
                        for path in images[start:end]
                    }

                    for future in as_completed(future_results):
                        image_path = future_results[future]
                        try:
                            result, boxes = future.result()
                        except:
                            print(image_path)
                            raise
                        bar.update()

                        if args.save_boxes:
                            all_boxes.append(boxes)
                            all_images.append(image_path)

            if args.save_boxes and all_boxes:
                np.savez(save_boxes, **dict(zip(all_images, all_boxes)))
