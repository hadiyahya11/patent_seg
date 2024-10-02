import os
import cv2
import numpy as np
import json
import sys
import pathlib

ALREADY_PREDICTED = "images_with_predictions.txt"

def is_contour_inside(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)  # inside
    x2, y2, w2, h2 = cv2.boundingRect(contour2)  # outside
    if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
        return True

def bounding_box_to_contour(box):
    x, y, w, h = box
    contour = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])
    return contour

def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 0, 150)
    kernel = np.ones((5, 5), np.uint8)
    img_dilate = cv2.dilate(img_canny, kernel, iterations=3)
    return img_dilate

def first_phase(img_path):
    img = cv2.imread(img_path)
    img_copy = np.copy(img)

    processed_img = process(img_copy)
    contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours = [ct for ct in contours if cv2.contourArea(ct) > 3000] #filter out small contours
    areas = []
    original_size = img_copy.shape
    for ct in filter_contours:
        x, y, w, h = cv2.boundingRect(ct)
        areas.append(w*h)
    biggest = filter_contours[areas.index(max(areas))]
    oldx, oldy, oldw, oldh = cv2.boundingRect(biggest)

    paddingx= 100
    paddingy= 100
    new_w = oldw - paddingx
    new_h = oldh - paddingy
    new_x = oldx + paddingx
    new_y = oldy + paddingy
    biggest = img_copy[new_y:oldy+new_h, new_x:oldx+new_w]
    # Create a white background image with the same size as the original
    white_background = np.ones((original_size[0], original_size[1], 3), dtype=np.uint8) * 255

    white_background[new_y:new_y+biggest.shape[0], new_x:new_x+biggest.shape[1]] = biggest

    return img_copy, filter_contours, white_background

def second_phase(img_first_phase, img_original):
    img_copy = np.copy(img_first_phase)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(process(img_copy), connectivity=8, ltype=cv2.CV_32S)
    contours = []
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        box = (x, y, w, h)
        contour = bounding_box_to_contour(box)
        contours.append(contour)

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 3000]
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

    for idx, contour in enumerate(sorted_contours):
        if idx == 0:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            res = False
            for i in range(idx):
                if is_contour_inside(contour, sorted_contours[i]):
                    res = True
                    break
            if not res:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img_copy, sorted_contours, img_original

def segment(img_path):
    img_copy, _, figure = first_phase(img_path)
    _, contours, segmented = second_phase(figure, img_copy)

    return contours, segmented

def to_json(contours, image_path, port):
    image_path = pathlib.Path(image_path)
    img = cv2.imread(image_path)
    image_height, image_width, _ = img.shape
    results = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        x_percent = np.round((x / image_width) * 100, 2)
        y_percent = np.round((y / image_height) * 100, 2)
        width_percent = np.round((w / image_width) * 100, 2)
        height_percent = np.round((h / image_height) * 100, 2)

        r = {
            "id": "result" + str(i),
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": 600,
            "original_height": 403,
            "image_rotation": 0,
            "value": {
                "rotation": 0,
                "x": x_percent,
                "y": y_percent,
                "width": width_percent,
                "height": height_percent,
                "rectanglelabels": ["contour"]
            }
        }
        results.append(r)

    image_name = image_path.with_suffix('.png').name
    image_path_json = f"http://localhost:{port}/{image_name}"

    json_entry = {
        "data": {
            "image": image_path_json
        },
        "predictions": [
            {
                "model_version": "one",
                "score": 0.5,
                "result": results
            }
        ]
    }

    return json_entry


if __name__ == "__main__":
    import argparse
    import datetime
    from tqdm import tqdm

    NOW = datetime.datetime.now(
        datetime.timezone(
            datetime.timedelta(hours=2)
        )
    ).strftime("%Y-%m-%dT%H%M")

    parser = argparse.ArgumentParser("Predict bounding boxes of segments in images")
    parser.add_argument("--take-from",
                        help="Take first or last images of directory",
                        choices=["start", "end"],
                        default="start")
    parser.add_argument("--directory",
                        default="dataset-dev/plates/",
                        type=pathlib.Path)
    parser.add_argument("-p", "--port",
                        help="port on which images will be served for LayoutParser (default: 8081)",
                        default=8081,
                        type=int)
    parser.add_argument("--prefix",
                        help='only parse files starting with prefix (default: "GB.")',
                        default="GB.")
    parser.add_argument("-o", "--output",
                        help="path for JSON file containing predicted boxes",
                        default=f"{NOW}_predictions.json")
    parser.add_argument("num_images",
                        type=int,
                        help="number of images for which to predict boxes")
    args = parser.parse_args()

    num_images = args.num_images
    plates = sorted([img for img in args.directory.iterdir() if img.name.startswith(args.prefix)])
    plates = [plate.resolve().relative_to(pathlib.Path.cwd()) for plate in plates]

    segmented_images = []
    try:
        with open(ALREADY_PREDICTED, 'r') as f:
            segmented_images = f.readlines()
        segmented_images = [img.strip() for img in segmented_images]
    except FileNotFoundError:
        with open(ALREADY_PREDICTED, 'w') as f:
            pass

    # remove the images that have already been segmented
    plates = [plate for plate in plates if plate.as_posix() not in segmented_images]

    if args.take_from == "start":
        selected_images = plates[:num_images]
    else:
        selected_images = plates[-num_images:]

    json_output = []
    images_with_predictions = []
    for img_path in tqdm(selected_images):
        contours, segmented = segment(img_path.as_posix())
        entry = to_json(contours, img_path, args.port)
        json_output.append(entry)
        images_with_predictions.append(
            img_path.resolve().relative_to(pathlib.Path.cwd()).as_posix()
        )

    with open(args.output, 'w') as jsonfile:
        json.dump(json_output, jsonfile, indent=2)

    # only save once all have been predicted
    with open(ALREADY_PREDICTED, 'a') as f:
        f.write('\n'.join(images_with_predictions))
        f.write('\n')

    print(f"JSON output saved to {args.output}")
