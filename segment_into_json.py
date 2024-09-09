import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import json

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


#first phase of segmentation to remove outer border present in images
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

    return img_copy, filter_contours,white_background
        
    
def second_phase(img_first_phase,img_original):
    img_copy = np.copy(img_first_phase)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(process(img_copy), connectivity=8, ltype=cv2.CV_32S)
    contours = []
    #area = img_copy.shape[0] * img_copy.shape[1]
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        box = (x, y, w, h)
        contour = bounding_box_to_contour(box)
        contours.append(contour)
        
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 3000]
    # get rid of small contours
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
    
        
    for idx,contour in enumerate(sorted_contours):
        if idx == 0:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            #x, y, w, h = cv2.boundingRect(contour)
            res = False
            for i in range(idx):
                if is_contour_inside(contour, sorted_contours[i]):
                    res = True
                    break
            if not res: #the contour is not inside any of the previous contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
    return img_copy, sorted_contours,img_original
    

def segment(img_path):
    img_copy, _,figure = first_phase(img_path)
    _, contours,segmented = second_phase(figure,img_copy)

    return contours,segmented

#convert to json format suitable for Label-studio
def to_json(contours,image_path):
    img = cv2.imread(image_path)
    image_height, image_width, _ = img.shape
    results = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Convert coordinates to percentages relative to the image size
        x_percent = np.round((x / image_width) * 100,2)
        y_percent = np.round((y / image_height) * 100,2)
        width_percent = np.round((w / image_width) * 100,2)
        height_percent = np.round((h / image_height) * 100,2)
        
        # Create the bounding box entry
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
    
    
    image_path_json = image_path[image_path.find('GB'):]
    image_path_json = image_path_json.replace('.tif', '.png')
    image_path_json = "http://localhost:8081/" + image_path_json
    # Structure the JSON entry for this row
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
    json_output = []
    #get the path from the user
    data_path = 'dataset-dev/plates/'
    gb_plates = [os.path.join(data_path, _pl) for _pl in os.listdir(data_path) if _pl.startswith("GB.")]
    
    for img_path in gb_plates[200:300]:
        contours, segmented = segment(img_path)
        entry = to_json(contours,img_path)
        json_output.append(entry)
        
        
    json_file_path = 'new_dataset_ml.json'
    # Write the resulting JSON to a file
    with open(json_file_path, 'w') as jsonfile:
        json.dump(json_output, jsonfile, indent=2)

    print(f"JSON output saved to {json_file_path}")

    