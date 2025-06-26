# Data Conventions

## Camera Coordinate Conventions

We follow the **COLMAP/OpenCV** camera coordinate convention:

- **+X** axis points to the right  
- **+Y** axis points downward  
- **+Z** axis points forward  

Note: This differs from the **OpenGL/Blender** convention, where the **Y** and **Z** axes are flipped, but **+X** remains the same.

## Dataset Format

The dataset format in `scrstudio` follows the structure used in [DSAC*](https://github.com/vislearn/dsacstar#data-structure) but with some adjustments for performance: rather than storing each image and pose as separate small files, we pack the images, intrinsics, and extrinsics into a few consolidated files.

Each scene is organized into self-contained folders for each split (`train`, `test`, `val`, etc.), with the following structure inside each split:

```
scene_name/
  ├── split_name/
  │     ├── rgb/             # raw image files (any image format supported by imageio)
  │     ├── rgb_lmdb/        # alternative to rgb/, storing images in LMDB format
  │     ├── poses.npy        # camera-to-world 4x4 pose matrices
  │     ├── calibration.npy  # camera intrinsics
```

### Image Folders

- `rgb/`: Contains raw image files in any format supported by `imageio`.  
- `rgb_lmdb/`: Images stored in LMDB format. The associated `file_list.txt` maps image indices to original relative paths (sorted alphabetically). The LMDB key is the image index (as in `file_list.txt`), and the value is the raw image file bytes.

### Pose File (`poses.npy`)

- A single `.npy` file containing array of **4×4 camera-to-world transformation matrices**.
- Ordered by the alphabetical sorting of image paths.

### Calibration File (`calibration.npy`)

- A single `.npy` file specifying intrinsics:
  - Either a scalar focal length (shared across x and y), or two values `(fx, fy)`, or a full **3×3 intrinsic matrix**.
  - The principal point is assumed to be at the image center unless a full **3×3 intrinsic matrix** is provided per image.
- Ordered by the alphabetical sorting of image paths.


## Convertion from COLMAP

We also provide a script to process COLMAP reconstructions into the `scrstudio` format. The script will undistort images, extract poses, and save them in the required structure. The script can be run as follows:

```bash
python scrstudio/scripts/process_data.py --sfm_path /path/to/colmap/sfm --image_path /path/to/images --output_path /path/to/output
```