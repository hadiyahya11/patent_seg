import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Get data that I want to segment

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

# First phase of segmentation to remove outer border present in images
def first_phase(img_path):
    img = cv2.imread(img_path)
    img_copy = np.copy(img)
    
    processed_img = process(img_copy)
    contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours = [ct for ct in contours if cv2.contourArea(ct) > 3000]  # filter out small contours
    areas = []
    original_size = img_copy.shape
    for ct in filter_contours:
        x, y, w, h = cv2.boundingRect(ct)
        areas.append(w * h)
    biggest = filter_contours[areas.index(max(areas))]
    oldx, oldy, oldw, oldh = cv2.boundingRect(biggest)

    paddingx = 100
    paddingy = 100
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
    # area = img_copy.shape[0] * img_copy.shape[1]
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        box = (x, y, w, h)
        contour = bounding_box_to_contour(box)
        contours.append(contour)
        
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 3000]
    # get rid of small contours
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
            if not res:  # the contour is not inside any of the previous contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img_copy, sorted_contours, img_original

def segment(img_path):
    img_copy, _, figure = first_phase(img_path)

    _, contours, segmented = second_phase(figure, img_copy)

    return contours, segmented

def on_button_click(event, path, contours, data, plt_fig, button_data):
    if event.inaxes == button_data['valid_button'].ax:
        valid = 1
    elif event.inaxes == button_data['reject_button'].ax:
        valid = 0
    
    # Append the result to the existing DataFrame
    data.loc[len(data)] = [path, contours, valid]
    
    # Save the results to a CSV file
    data.to_csv('/Users/mac/Desktop/patent_seg/segment.csv', index=False)
    
    # Close the current plot window and stop event loop
    plt_fig.canvas.stop_event_loop()
    plt.close(plt_fig)
    
    # Move to the next image if any
    button_data['next_image']()

def annotate(data):
    gb_paths_iter = iter(gb_paths)

    def next_image():
        try:
            path = next(gb_paths_iter)
            # Perform segmentation on the next image
            contours, segmented_image = segment(path)

            # Display the segmented image using matplotlib
            plt_fig, ax = plt.subplots(figsize=(15, 8))
            ax.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
            plt.title('Segmented Image')
            plt.axis('off')

            # Create valid and reject buttons
            valid_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
            reject_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
            button_data = {
                'valid_button': Button(valid_ax, 'Valid'),
                'reject_button': Button(reject_ax, 'Reject'),
                'next_image': next_image
            }
            
            button_data['valid_button'].on_clicked(lambda event: on_button_click(event, path, contours, data, plt_fig, button_data))
            button_data['reject_button'].on_clicked(lambda event: on_button_click(event, path, contours, data, plt_fig, button_data))

            plt.show(block=True)
        except StopIteration:
            print("Annotation process completed.")
            # Save the final DataFrame to the CSV file
            data.to_csv('/Users/mac/Desktop/patent_seg/segment.csv', index=False)

    # Start the annotation with the first image
    next_image()

# Execute the annotation process
if __name__ == "__main__":
    dev_plates_dataset = './dataset-dev/plates'
    
    # Create a CSV file with the columns Image, Contours, valid if it does not exist
    if not os.path.exists('/Users/mac/Desktop/patent_seg/segment.csv'):
        results = pd.DataFrame(columns=['Image', 'Contours', 'valid'])
    else:
        results = pd.read_csv('/Users/mac/Desktop/patent_seg/segment.csv')
    
    # Load all image paths in the dataset
    all_image_paths = [os.path.join(dev_plates_dataset, _pl) for _pl in os.listdir(dev_plates_dataset) if _pl.startswith("GB.")]
    
    # Filter out images that have already been validated
    validated_images = set(results['Image'].values)
    gb_paths = [img_path for img_path in all_image_paths if img_path not in validated_images]
    
    # Annotate the remaining images one by one
    if gb_paths:
        annotate(results)
    else:
        print("All images have already been validated.")
