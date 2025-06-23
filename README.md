# Floripa Guessr

A deep learning application to guess Florianópolis locations from images.

## About

Floripa Guessr is a neural network made to identify and classify locations in Florianópolis through images.

## Results

The final model achieves **61% accuracy** in location classification. 

Most classification errors occur between adjacent cells, indicating the model successfully learns geographic patterns but struggles with precise boundaries between neighboring areas.

## Project Structure

```
floripa-guessr/
│
├── dataset/
│   ├── images/                       # Downloaded images
│   ├── downloader.py                 # Google Street View image extractor
│   ├── csv_cleaner.py                # Dataset cleaning utility
│   ├── preparer.py                   # Dataset preparation script
│   ├── cell_visualizer/
│   |   └── grid_overlay.py           # Grid visualization utility
|   └── manifests/
│       ├── manifest.csv              # Main image manifest
│       ├── grid_bounds.csv           # Cell boundaries definition
│       ├── training_manifest.csv     # Training dataset manifest
│       └── validation_manifest.csv   # Validation dataset manifest
│
├── models/                           # Neural network checkpoints and exports
├── trainer.ipynb                     # Training notebook
```

Note that manifest files are "gitignored" and created at runtime.

## Scripts

### [downloader](dataset/downloader.py)
Panoramic image extractor using Google Street View API. Downloads images to `dataset/images/` and creates a manifest in `dataset/manifests/manifest.csv` containing image coordinates and file paths. The Google API must be externalized as an environment variable.

### [csv_cleaner](dataset/csv_cleaner.py)
Utility script that removes:
- Images not present in the manifest file
- Manifest entries without valid image paths

### [preparer](dataset/preparer.py)
Prepares the dataset by creating classes by grouping images by geographic proximity. This is done using KMeans clustering, balancing sample density per class/cell to prevent high imbalance rates. Outputs results to:
  - `dataset/manifests/grid_bounds.csv` - cells and their boundaries
  - `dataset/manifests/training_manifest.csv` - training set with file paths and assigned classes/cells
  - `dataset/manifests/validation_manifest.csv` - validation set with file paths and assigned classes/cells

### [grid_overlay](dataset/cell_visualizer/grid_overlay.py)
Utility script that overlays cell boundaries from `dataset/manifests/grid_bounds.csv` onto a map image for easy visualization of the class representation created by the dataset preparation script. Requires:
- Map image of the region
- Map boundary coordinates (currently hard coded)

### [trainer](trainer.ipynb)
Jupyter notebook that:
- Defines and creates the neural network for image classification
- Uses fast.ai framework with EfficientNet-B4 architecture
- Performs image preprocessing (transforms) before training
- Saves checkpoints and exports to `models/` directory

## Models

You can access the exported models and weights in the releases.