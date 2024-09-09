import marimo

__generated_with = "0.8.7"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import os
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return cv2, np, os, pd, plt


@app.cell
def __(cv2, np):
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
    return bounding_box_to_contour, is_contour_inside, process


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(bounding_box_to_contour, cv2, is_contour_inside, np, process):
    def first_phase(img_path):
        img = cv2.imread(img_path)
        img_copy = np.copy(img)

        processed_img = process(img_copy)
        contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filter_contours = [ct for ct in contours if cv2.contourArea(ct) > 3000] # filter out small contours
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
        biggest = img_copy[new_y:oldy + new_h, new_x:oldx + new_w]
        # Create a white background image with the same size as the original
        white_background = np.ones((original_size[0], original_size[1], 3), dtype=np.uint8) * 255

        white_background[new_y:new_y + biggest.shape[0], new_x:new_x + biggest.shape[1]] = biggest

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
    return first_phase, second_phase, segment


@app.cell
def __():
    OUTFILE = 'seg_nc_2.csv'
    return OUTFILE,


@app.cell
def __(OUTFILE, os, pd):
    dev_plates_dataset = './dataset-dev/plates'
    # Create a CSV file with the columns Image, Contours, valid if it does not exist
    results = pd.DataFrame()
    if not os.path.exists(OUTFILE):
        results = pd.DataFrame(columns=['Image','Contours','valid'])
    else:
        results = pd.read_csv(OUTFILE)

    gb_paths = [os.path.join(dev_plates_dataset, _pl) for _pl in 
    os.listdir(dev_plates_dataset) if (_pl.startswith("GB."))]
    validated = set(results['Image'].values)
    if len(results)!=0:
        gb_paths = [path for path in gb_paths if path not in validated]
    return dev_plates_dataset, gb_paths, results, validated


@app.cell
def __(mo):
    # Validate button
    validate = mo.ui.run_button(
                label="Validate",
                #on_click=lambda _: on_validate(index, path, contours, results)
                #value=0, 
                #on_click=lambda value: value + 1,
                #on_change= lambda _: 
            )

            # Reject button
    reject = mo.ui.run_button(
                label="Reject",
                #on_click=lambda: on_reject(index, path, contours, results)
            )
    return reject, validate


@app.cell
def __(mo, process_image, reject, validate, validated):
    def display(results):
        contours, segmented_image, path = process_image(results)
        if path is None:
            return  # Stop if no more images to process

        mo.output.append(mo.plain_text(f"{len(validated)} {path}"))
        mo.output.append(mo.image(segmented_image, width=800))
        mo.output.append(validate)
        mo.output.append(reject)

        #return path, contours  # Return the path and contours for further use
    return display,


@app.cell
def __(gb_paths, segment, validated):
    def process_image(results):
        try:
            path = next(path for path in gb_paths if path not in validated)
        except StopIteration:
            print("Finished processing all images.")
            return None, None, None

        contours, segmented_image = segment(path)
        print(contours)

        results.loc[len(results)] = [path, contours, int(1)]
        return contours, segmented_image, path
    return process_image,


@app.cell
def __(display, results):
    display(results)
    return


@app.cell
def __(OUTFILE, mo, reject, results, validate, validated):
    mo.stop(not(validate.value or reject.value))
    validated.add(results.iloc[-1]['Image'])
    if validate.value:
        pass
    elif reject.value:
        #access last element of results
        results.loc[results.index[-1], 'valid'] = int(0)

    results.to_csv(OUTFILE, index=False)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
