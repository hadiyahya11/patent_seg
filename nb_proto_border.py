import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import pathlib
    import numpy as np
    import cv2
    return cv2, np, pathlib


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(cv2, np):
    def first_phase(img_path, keep_n_largest=3, dilation_kernel=(5, 5)):
        img = cv2.imread(img_path)
        img_copy = np.copy(img)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(img_gray, 0, 150)
        kernel = np.ones(dilation_kernel, np.uint8)
        img_dilate = cv2.dilate(img_canny, kernel, iterations=3)

        contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filter_contours = [ct for ct in contours if cv2.contourArea(ct) > 3000] #filter out small contours

        # only keep the largest contours
        areas = []
        for ct in filter_contours:
            areas.append(cv2.contourArea(ct))
        indices = np.flip(np.argsort(areas))[:keep_n_largest]
        contours = [filter_contours[ix] for ix in indices]

        return img_copy, contours
    return (first_phase,)


@app.cell
def __(cv2, first_phase, mo):
    _im, _c = first_phase("dataset-dev/preprocessed_alt/GB.109315.A-004.png")
    mo.image(cv2.drawContours(_im, _c, -1, (0, 255, 0), 3))
    return


@app.cell
def __(cv2, first_phase, mo):
    _im, _c = first_phase("dataset-dev/plates_png/GB.302614.A-023.png")
    mo.image(cv2.drawContours(_im, _c, -1, (0, 255, 0), 3))
    return


@app.cell
def __(mo):
    run = mo.ui.run_button()
    run
    return (run,)


@app.cell
def __(cv2, first_phase, mo, pathlib, run):
    mo.stop(not run.value)

    _n = 50
    _i = 0

    for _plate in pathlib.Path("dataset-dev/plates_png/").rglob("GB.*.png"):
        _i += 1
        if _i == _n:
            break

        _im, _c = first_phase(_plate) 
        mo.output.append(mo.plain_text(_plate))
        mo.output.append(mo.image(cv2.drawContours(_im, _c, -1, (0, 255, 0), 3)))
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
