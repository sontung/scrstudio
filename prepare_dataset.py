import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

import lmdb
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from io import BytesIO
import imagesize
import quaternion
from scipy.spatial.transform import Rotation

from scrstudio.data.utils.readers import LMDBReaderConfig


CONDITIONS = [
    "dawn",
    "dusk",
    "night",
    "night-rain",
    "overcast-summer",
    "overcast-winter",
    "rain",
    "snow",
    "sun",
]


def read_train_poses(a_file, cl=False):
    with open(a_file) as file:
        lines = [line.rstrip() for line in file]
    if cl:
        lines = lines[4:]
    name2mat = {}
    for line in lines:
        img_name, *matrix = line.split(" ")
        if len(matrix) == 16:
            matrix = np.array(matrix, float).reshape(4, 4)
        name2mat[img_name] = matrix
    return name2mat


class RobotCarDataset(Dataset):
    images_dir_str: str

    def __init__(self, ds_dir="datasets/robotcar", train=True, evaluate=False):
        ds_type = "robotcar"
        ds_dir = ds_dir
        sfm_model_dir = f"{ds_dir}/3D-models/all-merged/all.nvm"
        images_dir = Path(f"{ds_dir}/images")
        # test_file1 = f"{ds_dir}/robotcar_v2_train.txt"
        test_file2 = f"{ds_dir}/robotcar_v2_test.txt"
        ds_dir_path = Path(ds_dir)
        images_dir_str = str(images_dir)
        train = train
        evaluate = evaluate
        if evaluate:
            assert not train

        if train:
            (
                xyz_arr,
                image2points,
                image2name,
                image2info,
                image2uvs,
                image2pose,
            ) = read_nvm_file(sfm_model_dir)
            name2image = {v: k for k, v in image2name.items()}
            img_ids = list(image2name.keys())

        else:
            ts2cond = {}
            for condition in CONDITIONS:
                all_image_names = list(Path.glob(images_dir, f"{condition}/*/*"))

                for name in all_image_names:
                    time_stamp = str(name).split("/")[-1].split(".")[0]
                    ts2cond.setdefault(time_stamp, []).append(condition)
            for ts in ts2cond:
                assert len(ts2cond[ts]) == 3

            name2mat = read_train_poses(test_file2)
            img_ids = list(name2mat.keys())

        return

    def _process_id_to_name(self, img_id):
        name = image2name[img_id].split("./")[-1]
        name2 = str(images_dir / name).replace(".png", ".jpg")
        return name2

    def __len__(self):
        return len(img_ids)

    def _get_single_item(self, idx):
        if train:
            img_id = img_ids[idx]
            image_name = _process_id_to_name(img_id)

        else:
            name0 = img_ids[idx]

            time_stamp = str(name0).split("/")[-1].split(".")[0]
            cond = ts2cond[time_stamp][0]
            name1 = f"{cond}/{name0}"
            if ".png" in name1:
                name1 = name1.replace(".png", ".jpg")

            image_name = str(images_dir / name1)
        return image_name

    def __getitem__(self, idx):

        return _get_single_item(idx)


def folder2lmdb(image_folder, lmdb_path):
    image_paths = glob(os.path.join(image_folder, '**', '*.*'), recursive=True)
    image_paths = [f for f in image_paths if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_paths.sort()
    lmdb_path = str(lmdb_path)
    if not os.path.isdir(lmdb_path):
        os.makedirs(lmdb_path)
    else:
        shutil.rmtree(lmdb_path)
        print('lmdb path exist, remove it')
        os.makedirs(lmdb_path)
    with open(os.path.join(lmdb_path, 'file_list.txt'), 'w') as f:
        for image_path in image_paths:
            # remove image_folder from image_path
            image_path = os.path.relpath(image_path, image_folder)
            f.write(image_path+'\n')
    env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                         map_size=2**36, readonly=False,
                         meminit=False, map_async=True,max_dbs=1)
    db=env.open_db(b'images',integerkey=True)
    with env.begin(write=True, db=db) as txn:
        for i, image_path in enumerate(tqdm(image_paths)):
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            txn.put(key=i.to_bytes(4,sys.byteorder), value=image_bytes,append=True)
    env.close()


def parse_nvm_intrinsics(all_names, name_id, output_path, image_folder):
    lines = []
    fx, fy = [362.39892808341875, 365.24253793162524]
    cx, cy = [482.2320501297286, 289.9148760046266]
    path_rest = []
    for name in tqdm(all_names):
        h, w = imagesize.get(name)
        name2 = str(name).split(image_folder)[-1][1:]
        out_line = f"{name2} PINHOLE {h} {w} {fx} {fy} {cx} {cy}"
        lines.append(out_line)
        path_rest.append("/".join(name2.split("/")[:-1]))
    with open(f"{output_path}/intrinsics_{name_id}.txt", "w") as f:
        f.write("\n".join(lines))
    return set(path_rest)


def run_command(cmd: str, verbose=False):
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    return out


def read_lines_from_file(filepath):
    """Read all lines from a text file and return them as a list."""
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip().split(" ")[0] for line in lines]  # Remove trailing newlines


def lmdb_image_shapes(path):
    path=Path(path)
    reader=LMDBReaderConfig(img_type='bytes',db_name="images").setup(root=path)
    image_shapes = []
    for i in tqdm(range(len(reader))):
        bio=BytesIO(reader[i])
        w,h=imagesize.get(bio)
        image_shapes.append((h,w))
    image_shapes=np.stack(image_shapes)
    np.save(path/'image_shapes.npy',image_shapes)

def quat_to_matrix(tx_, ty_, tz_, qw_, qx_, qy_, qz_):
    """Convert translation + quaternion → 4×4 SE(3) matrix using scipy."""
    # scipy quaternion order = (x, y, z, w)
    rot = Rotation.from_quat([qx_, qy_, qz_, qw_])
    Rmat = rot.as_matrix()

    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = [tx_, ty_, tz_]
    return T

def read_evo_pose_file(txt_dir):
    pose_mats = []
    all_ts = []
    ts2pose = {}
    with open(txt_dir, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            tx, ty, tz, qx, qy, qz, qw = list(map(float, line.split()[1:]))
            ts = line.split()[0]
            pose_mat = quat_to_matrix(tx, ty, tz, qw, qx, qy, qz)
            # pose_mat = np.linalg.inv(pose_mat)
            pose_mats.append(pose_mat)
            all_ts.append(str(ts))
            ts2pose[str(ts)] = pose_mat

    return all_ts, pose_mats, ts2pose

def read_image_names_and_poses(root_dirs, image_folder):
    rgb_files = []
    all_poses = []
    for root_dir in root_dirs:
        root_dir = Path(image_folder) / Path(root_dir)

        # Main folders.
        rgb_dir = root_dir / "cam0"
        pose_dir = root_dir / "poses.txt"

        ts_ids, _, ts2pose = read_evo_pose_file(pose_dir)
        all_names = sorted(rgb_dir.iterdir())
        count = 0
        for ts_id in ts_ids:
            im_name = rgb_dir/f"{ts_id}000.png"
            if im_name in all_names:
                count += 1
                rgb_files.append(im_name)
                all_poses.append(ts2pose[ts_id])
    assert len(rgb_files) == len(rgb_files), f"{len(rgb_files)} {len(rgb_files)}"
    return rgb_files, all_poses


def main(train_paths=[], test_paths=[], output_path=""):
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=True)

    image_folder = "/home/vr/work/datasets"
    train_images, train_poses = read_image_names_and_poses(train_paths, image_folder)
    test_images, test_poses = read_image_names_and_poses(test_paths, image_folder)

    path_rest1 = parse_nvm_intrinsics(train_images, "train", output_path, image_folder)
    path_rest2 = parse_nvm_intrinsics(test_images, "test", output_path, image_folder)
    print(len(train_images), len(train_poses))
    # sys.exit()
    for r in path_rest1:
        r2 = root / "train/rgb" / r
        r2.mkdir(parents=True, exist_ok=True)
    for r in path_rest2:
        r2 = root / "test/rgb" / r
        r2.mkdir(parents=True, exist_ok=True)

    test_dir = Path(f"{root}/test")
    train_dir = Path(f"{root}/train")
    test_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    # cmd = f"colmap image_undistorter_standalone --input_file {output_path}/intrinsics_train.txt --image_path {image_folder} --output_path {root}/train/rgb"
    # run_command(cmd, verbose=True)
    # cmd = f"colmap image_undistorter_standalone --input_file {output_path}/intrinsics_test.txt --image_path {image_folder} --output_path {root}/test/rgb"
    # run_command(cmd, verbose=True)

    # create folders (colmap will not create them)
    folder2lmdb(test_dir / "rgb", test_dir / "rgb_lmdb")

    folder2lmdb(test_dir / "rgb", test_dir / "rgb_lmdb")
    folder2lmdb(train_dir / "rgb", train_dir / "rgb_lmdb")
    lmdb_image_shapes(train_dir)
    
    train_poses = np.stack(train_poses)
    np.save(train_dir / "poses.npy", train_poses)
    test_poses = np.stack(test_poses)
    np.save(test_dir / "poses.npy", test_poses)

    # placeholder
    calibrations = np.ones(train_poses.shape[0], dtype=np.float32)
    np.save(train_dir / "calibration.npy", calibrations)
    calibrations = np.ones(test_poses.shape[0], dtype=np.float32)
    np.save(test_dir / "calibration.npy", calibrations)


if __name__ == "__main__":
    # main(Path("datasets/robotcar"))
    main(train_paths=["zed_recording_data_20251127_train"],
         test_paths=["zed_recording_data_20251127_test_1", 
         "zed_recording_data_20251127_test_2"],
         output_path="/home/vr/work/datasets/vr_general")
