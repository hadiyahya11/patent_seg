import marimo

__generated_with = "0.7.17"
app = marimo.App(width="medium")


@app.cell
def __():
    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    return cv2, np, os, plt


@app.cell
def __(mo):
    mo.md(r"""## Testing on individual images""")
    return


@app.cell
def __():
    data_dir = "sample-2024-03-28/images/"
    return data_dir,


@app.cell
def __():
    #img_path = os.path.join(data_dir, os.listdir(data_dir)[0])
    return


@app.cell
def __(current):
    img_path = current['plate']
    #img_path = "dataset-dev/plates/FR.857973.A-000.tif"
    #img_path = "dataset-dev/plates/GB.190925559.A-011.tif"
    #img_path = "dataset-dev/plates/FR.816992.A-000.tif"
    img_path = "dataset-dev/plates/FR.324746.A-008.tif"
    #img_path = "dataset-dev/plates/GB.121298.A-009.tif"
    #img_path = test_plate
    #img_path = list(validated.keys())[-1]
    #img_path = "dataset-dev/plates/FR.411156.A-002.tif"
    return img_path,


@app.cell
def __(cv2, img_path, np, os, plt):
    assert img_path is not None and os.path.exists(img_path)

    im = cv2.imread(img_path)
    _im_copy = np.copy(im)

    _im_gray = cv2.cvtColor(_im_copy, cv2.COLOR_BGR2GRAY)
    _blur = cv2.GaussianBlur(_im_gray, (0,0), sigmaX=100, sigmaY=100)
    _divided = cv2.divide(_im_gray, _blur, scale=255)

    #_ret, _thresh = cv2.threshold(_im_gray, 50, 255, cv2.THRESH_BINARY)
    _im_canny = cv2.Canny(_divided, 0, 150)
    _kernel = np.ones((5, 5), np.uint8)
    im_dilate = cv2.dilate(_im_canny, _kernel, iterations=5)
    #im_dilate = _im_canny

    plt.imshow(im_dilate, 'gray')
    plt.show()
    return im, im_dilate


@app.cell
def __(cv2, get_contours, im, img_path, plt):
    _res_im = im.copy()

    _contours, _dropped = get_contours(img_path, include_dropped=True)

    cv2.drawContours(_res_im, _contours, -1, (0,255,0), 3)
    for _c in _dropped:
        if _c[1] == 'small':
            cv2.drawContours(_res_im, [_c[0]], 0, (255,0,0), 3)
        elif _c[1] == 'russian-doll':
            cv2.drawContours(_res_im, [_c[0]], 0, (0,0,255), 3)
        elif _c[1] == 'large':
            cv2.drawContours(_res_im, [_c[0]], 0, (255,0,255), 3)
        elif _c[1] == 'container':
            cv2.drawContours(_res_im, [_c[0]], 0, (255,255,0), 3)
        elif _c[1] == 'outside-large':
            cv2.drawContours(_res_im, [_c[0]], 0, (255,0,255), 3)
        elif _c[1] == 'large1':
            cv2.drawContours(_res_im, [_c[0]], 0, (30,30,30), 3)
        elif _c[1] == 'narrow':
            cv2.drawContours(_res_im, [_c[0]], 0, (255,150,0), 3)

    plt.imshow(_res_im)
    plt.show()

    _contours
    _dropped
    return


@app.cell
def __(cv2, im, im_dilate, np, plt):
    _num_labels, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(im_dilate, connectivity=4, ltype=cv2.CV_32S)

    _total_area = im_dilate.shape[0] * im_dilate.shape[1]
    _min_area = _total_area // 150
    print(_min_area)

    _contours = []
    for _i in range(1, _num_labels):
        _x, _y, _w, _h, _ = _stats[_i]
        _contour = np.array([
            [_x, _y],
            [_x + _w, _y],
            [_x + _w, _y + _h],
            [_x, _y + _h]
        ])
        _contours.append(_contour)

    _contours = [_c for _c in _contours if cv2.contourArea(_c) > _min_area]

    _res_im = im.copy()

    cv2.drawContours(_res_im, _contours, -1, (0,255,0), 3)
    plt.imshow(_res_im)
    plt.show()
    return


@app.cell
def __(cv2, im, im_dilate, plt):
    _res_im = im.copy()
    _im = cv2.fastNlMeansDenoising(im_dilate)
    _contours, _hierarchy = cv2.findContours(_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(_res_im, _contours, -1, (0,255,0), 3)
    plt.imshow(_res_im)
    plt.show()

    print(len(_contours))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Develop function iteratively""")
    return


@app.cell
def __():
    import time
    return time,


@app.cell
def __(cv2):
    def is_contour_inside(contour1, contour2):
        x1, y1, w1, h1 = cv2.boundingRect(contour1)  # inside
        x2, y2, w2, h2 = cv2.boundingRect(contour2)  # outside
        if x1 >= x2-5 and y1 >= y2-5 and x1+w1 <= x2+w2+5 and y1+h1 <= y2+h2+5:
            return True
    return is_contour_inside,


@app.cell
def __(cv2, is_contour_inside, np, os):
    def get_contours(img_path, include_dropped=False):
        #print(img_path)
        if not os.path.exists(img_path):
            return None

        _im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        #_im_gray = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
        _im_canny = cv2.Canny(_im, 0, 150)
        _kernel = np.ones((5, 5), np.uint8)
        _im_dilate = cv2.dilate(_im_canny, _kernel, iterations=5)

        _num_labels, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(
            _im_dilate, 
            connectivity=4,
            ltype=cv2.CV_32S
        )

        _total_area = _im_dilate.shape[0] * _im_dilate.shape[1]
        _min_area = _total_area // 175

        _contours = []
        for _i in range(_num_labels):
            _x, _y, _w, _h, _ = _stats[_i]
            _contour = np.array([
                [_x, _y],
                [_x + _w, _y],
                [_x + _w, _y + _h],
                [_x, _y + _h]
            ])
            _contours.append(_contour)

        _dropped = []

        # remove segments that are very narrow
        _minimum_ratio = 0.15
        _ratios = []
        for _i in range(_num_labels):
            _x, _y, _w, _h, _ = _stats[_i]
            _ratios.append(
                min(_w, _h) / (_w + _h)
            )
        _dropped += [
            (_c, 'narrow') for _i, _c in enumerate(_contours)
            if _ratios[_i] < _minimum_ratio
        ]
        _contours = [
            _c for _i, _c in enumerate(_contours)
            if _ratios[_i] >= _minimum_ratio
        ]

        # remove contour identical with image
        _contours = [_c for _c in _contours if tuple(_c[0]) != (0, 0) and tuple(_c[2]) != tuple(_im.shape)]

        # remove contours that are too small
        _dropped += [(_c, 'small') for _c in _contours if cv2.contourArea(_c) < _min_area]
        _contours = [_c for _c in _contours if cv2.contourArea(_c) >= _min_area]

        _contours = sorted(_contours, key=cv2.contourArea, reverse=True)

        # drop contours that are > 70% of image
        if len(_contours) > 1:
            _dropped += [(_c, 'large1') for _c in _contours if cv2.contourArea(_c) > _total_area * 0.7]
            _contours = [_c for _c in _contours if cv2.contourArea(_c) <= _total_area * 0.7]

        # russian dolls
        _to_drop = []
        _hierarchy = []
        for i, outside_contour in enumerate(_contours):
            inside = []
            for ii, contour2 in enumerate(_contours):
                if ii == i:
                    continue
                if is_contour_inside(contour2, outside_contour):
                    # only append direct children
                    if not any(is_contour_inside(contour2, _contours[iii]) for iii in inside):
                        inside.append(ii)
            _hierarchy.append((i, inside))
        #print(_hierarchy)
        for i, children in _hierarchy[::-1]:
            # no inner box or more than one, continue
            if len(children) != 1:
                continue

            ii = children[0]
            # inner box has many children
            #if len(_hierarchy[ii]) > 

            _area_parent = cv2.contourArea(_contours[i])
            _area_child = cv2.contourArea(_contours[ii])
            if _area_child / _area_parent < 0.1:
                # too small inner box, keep only outer box
                _to_drop.append(ii)
            elif _area_child / _area_parent > 0.45:
                # inner box fills outer box, keep only inner box
                _to_drop.append(i)
            else:
                print(_area_child / _area_parent)
                pass
        for ix in sorted(_to_drop, reverse=True):
            _dropped.append((_contours[ix], 'russian-doll'))
            _contours.pop(ix)

        if len(_contours) > 1:
            _copy = _contours[:]
            for i, c in enumerate(_contours):
                # if there are more than one other contour
                if len(_contours[i+1:]) < 2:
                    break
                # and if the current contour contains all the other ones…
                if all([is_contour_inside(cc, c) for cc in _contours[i+1:]]):
                    # and the inside contours cover at least 20% of area of current contour
                    # then drop it
                    #print(sum(cv2.contourArea(cc) for cc in _contours[i+1:]) / cv2.contourArea(c))
                    if sum(cv2.contourArea(cc) for cc in _contours[i+1:]) / cv2.contourArea(c) >= 0.2:
                        _dropped.append((_contours[i], 'container'))
                        _copy = _contours[i+1:]
            _contours = _copy

        if len(_contours) == 2:
            c1, c2 = _contours
            if cv2.contourArea(c2) < _total_area // 100 and is_contour_inside(c2, c1):
                _dropped.append((c2, 'inside-small'))
                _contours = [c1]

        if len(_contours) == 2:
            c1, c2 = _contours
            if cv2.contourArea(c1) / _total_area > 0.5 and is_contour_inside(c2, c1):
                _dropped.append((c1, 'outside-large'))
                _contours = [c2]

        # drop remaining contours that are > 60% of image
        if len(_contours) > 1:
            _dropped += [(_c, 'large2') for _c in _contours if cv2.contourArea(_c) > _total_area * 0.6]
            _contours = [_c for _c in _contours if cv2.contourArea(_c) <= _total_area * 0.6]

        if include_dropped:
            return np.array(_contours), _dropped
        else:
            return np.array(_contours)
    return get_contours,


@app.cell
def __():
    dev_plates_dataset = "dataset-dev/plates/"
    return dev_plates_dataset,


@app.cell
def __(dev_plates_dataset, os):
    fr_plates = [os.path.join(dev_plates_dataset, _pl) for _pl in os.listdir(dev_plates_dataset) if _pl.startswith("FR.")]
    gb_plates = [os.path.join(dev_plates_dataset, _pl) for _pl in os.listdir(dev_plates_dataset) if _pl.startswith("GB.")]
    validated = {}
    return fr_plates, gb_plates, validated


@app.cell
def __():
    current = {'plate': None, 'segments': None}
    return current,


@app.cell
def __(get_contours, time):
    _ = get_contours

    # whenever the cell defining get_contours is run, assume the function changed
    fn_version = time.time()
    return fn_version,


@app.cell(hide_code=True)
def __(mo):
    next_plate_from = mo.ui.radio(label="Take next plate from:", options=["fr_plates", "gb_plates"], value="fr_plates")
    iterate = mo.ui.run_button(label="Segment")
    mo.vstack([
        next_plate_from,
        iterate
    ])
    return iterate, next_plate_from


@app.cell
def __(
    compute_ratio,
    current,
    cv2,
    fn_version,
    fr_plates,
    gb_plates,
    get_contours,
    iterate,
    mo,
    next_plate_from,
    validated,
):
    validate = mo.ui.run_button(
        label="Validate",
        on_change=lambda _: mo.output.clear()
    )

    revalidate = mo.ui.run_button(
        label="Revalidate",
        on_change=lambda _: mo.output.clear()
    )

    if iterate.value:
        for test_plate, _old in validated.items():
            try:
                if current['version'] == fn_version:
                    continue
            except KeyError:
                pass

            tested_contours = get_contours(test_plate)
            if len(tested_contours) == len(_old) and (tested_contours != _old).any():
                _overlaps_proportions = []
                for _i, _c in enumerate(tested_contours):
                    # calculate overlap
                    _right_edge = min(_c[1][0], _old[_i][1][0])
                    _left_edge = max(_c[0][0], _old[_i][0][0])
                    _overlapping_x = _right_edge - _left_edge
                    _top_edge = min(_c[0][1], _old[_i][0][1])
                    _bottom_edge = max(_c[2][1], _old[_i][2][1])
                    _overlapping_y = _bottom_edge - _top_edge
                    _overlap_area = _overlapping_x * _overlapping_y
                    _overlaps_proportions.append(_overlap_area / cv2.contourArea(_c))
                    #mo.output.append(_overlap_area / cv2.contourArea(_c))
                    #mo.output.append(_overlap_area / cv2.contourArea(_old[_i]))

                #print(_overlaps_proportions)
                if all([_ov > 0.98 for _ov in _overlaps_proportions]):
                    continue

            if len(tested_contours) != len(_old) or (tested_contours != _old).any():
                _im = cv2.imread(test_plate)
                _im_old = _im.copy()
                cv2.drawContours(_im_old, _old, -1, (0,255,0), 3)
                _im_new = _im.copy()
                cv2.drawContours(_im_new, tested_contours, -1, (0,255,0), 3)
                for _i, _s in enumerate(tested_contours):
                    _x, _y = _s[0]
                    cv2.putText(_im_new, str(_i), (_x+20, _y+120), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255,0,0), 3)

                mo.output.append(
                    mo.hstack(
                        [
                            mo.image(_im_old, width=400, style={'border': '1px solid #000'}), 
                            mo.image(_im_new, width=400, style={'border': '1px solid #000'}),
                            mo.vstack([
                                revalidate,
                                mo.plain_text(f'{list(validated.keys()).index(test_plate)}: "{test_plate}"'),
                                list(_old),
                                [cv2.contourArea(_c) / (_im_old.shape[0] * _im_old.shape[1]) for _c in _old],
                                [compute_ratio(_c) for _c in _old],
                                list(tested_contours),
                                [cv2.contourArea(_c) / (_im_new.shape[0] * _im_new.shape[1]) for _c in tested_contours],
                                [compute_ratio(_c) for _c in tested_contours],
                            ])
                        ],
                        justify="start"
                    )
                )
                break
        else:
            _plate = current['plate']
            _segments = current['segments']

            if _plate is None:
                if next_plate_from.value == "fr_plates":
                    _plate = fr_plates.pop()
                if next_plate_from.value == "gb_plates":
                    _plate = gb_plates.pop()

            _segments = get_contours(_plate)

            current['plate'] = _plate
            current['segments'] = _segments

            _im = cv2.imread(_plate)
            cv2.drawContours(_im, _segments, -1, (0,255,0), 3)
            for _i, _s in enumerate(_segments):
                _x, _y = _s[0]
                cv2.putText(_im, str(_i), (_x+20, _y+120), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255,0,0), 3)

            _ratios = [compute_ratio(_s) for _s in _segments]
            _areas = [cv2.contourArea(_s) / (_im.shape[0] * _im.shape[1]) for _s in _segments]

            mo.output.append(
                mo.hstack(
                    [
                        mo.vstack([
                            mo.image(_im, width=400, style={'border': '1px solid #000'}),
                            validate
                        ]),
                        mo.vstack([
                            mo.plain_text(f'{len(validated)} validated'),
                            mo.plain_text(f'"{_plate}"'),
                            list(_segments),
                            mo.md(f"Areas: {mo.as_html(_areas)}"),
                            mo.md(f"Proportions: {mo.as_html(_ratios)}")
                        ])
                    ],
                    justify="start"
                )
            )
            current['version'] = fn_version
    return revalidate, test_plate, tested_contours, validate


@app.cell
def __(fr_plates):
    len(fr_plates)
    return


@app.cell(hide_code=True)
def __(get_contours, validate, validated):
    _f, _ran = get_contours, validate

    len(validated)
    return


@app.cell
def __(current, validate):
    _ran = validate

    current
    return


@app.cell
def __(current, mo, validate, validated):
    mo.stop(not validate.value)

    validated[current['plate']] = current['segments']
    current['plate'] = None
    current['segments'] = None
    mo.md("Reset!")
    return


@app.cell
def __(mo, revalidate, test_plate, tested_contours, validated):
    mo.stop(not revalidate.value)
    validated[test_plate] = tested_contours
    mo.md("Revalidated!")
    return


@app.cell
def __():
    def compute_ratio(contour):
        _pt1, _pt2, _pt3, _pt4 = contour
        _w = _pt2[0] - _pt1[0]
        _h = _pt3[1] - _pt1[1]
        return min(_w, _h) / (_w + _h)
    return compute_ratio,


@app.cell
def __():
    further_improvements = [
        "dataset-dev/plates/GB.495582.A-025.tif",
        "dataset-dev/plates/FR.830892.A-000.tif",
        "dataset-dev/plates/GB.232273.A-017.tif",
        "dataset-dev/plates/GB.319229.A-015.tif",
        "dataset-dev/plates/GB.125650.A-012.tif", # ou pas nécessaire car autres planches?
        "dataset-dev/plates/GB.191223716.A-012.tif",
        "dataset-dev/plates/GB.302614.A-018.tif"
    ]
    return further_improvements,


@app.cell
def __(mo):
    mo.md(r"""## Show segmenting on expected matches""")
    return


@app.cell
def __():
    good_matches = []
    with open("good-matches.txt") as f:
        for l in f:
            _gb, _fr = l.strip().split(",")
            good_matches.append((_fr, _gb))
    return f, good_matches, l


@app.cell
def __(good_matches):
    good_matches
    return


@app.cell
def __():
    import glob
    return glob,


@app.cell
def __(cv2, get_contours, glob, good_matches, mo):
    for _fr, _gb in good_matches:
        _segmented_plates = []
        for _plate in glob.iglob(f"dataset-dev/plates/{_fr}-*"):
            _segments = get_contours(_plate)
            _im = cv2.imread(_plate)
            cv2.drawContours(_im, _segments, -1, (0,255,0), 3)
            _segmented_plates.append(mo.image(_im, width=400, style={'border': '1px solid #000'}))
        for _plate in glob.iglob(f"dataset-dev/plates/{_gb}-*"):
            _segments = get_contours(_plate)
            _im = cv2.imread(_plate)
            cv2.drawContours(_im, _segments, -1, (0,255,0), 3)
            _segmented_plates.append(mo.image(_im, width=400, style={'border': '1px solid #000'}))
        mo.output.append(mo.hstack(_segmented_plates, justify="start"))
    return


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
