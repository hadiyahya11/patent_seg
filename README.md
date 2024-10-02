# Improving segmentation of patent drawings

## Annotating

To annotate data with LabelStudio:

- Transform the TIFF files into PNG files, for instance by using `tif_to_png.py`.
- Serve the data using a local server by running `local_files.py`, either
  directly in the directory where you want to serve files, or using the
  `--directory` option. (Optionally, you can change the port using the `--port`
  option).
- Run `predict_segments.py`, passing the number of files for which to predict
  bounding boxes (other options include `--port`, in case you changed the port
  above and `--take-from={start,end}` to process files from the start or the end
  of the list; use `predict_segments.py --help` for more options.
- Create a project in LabelStudio. In "Labeling Setup" select the template
  "Object Detection with Bounding Boxes", remove the default labels and create a
  new one called `contour`. In "Data Import" upload the JSON file created by
  `predict_segments.py`.
