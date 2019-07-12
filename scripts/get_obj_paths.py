#!/usr/bin/env python
import os
import yaml

SHAPENET_DIR = "/home/parker/datasets/ShapeNetCore.v2"
SUB_PATH = "models/model_normalized.obj"
PATHS_FILE = "../config/obj_paths.txt"
OUT_DIR = "/home/parker/datasets/ShapeNetPointCloud/"


def get_classes_from_yaml():
    classes = []
    with open("../config/shapenet_classes.yaml", "r") as f:
        try:
            data = yaml.load(f)
            for ref in data["classes"]:
                classes.append(ref)
        except yaml.YAMLError as e:
            print(e)

    return classes


if __name__ == '__main__':
    classes = get_classes_from_yaml()
    all_paths = []

    # Iterate through classes.
    for ref in classes:
        path = os.path.join(SHAPENET_DIR, ref["ref"])
        # Iterate through instances of the class.
        for filename in os.listdir(path):
            obj_path = os.path.join(path, filename, SUB_PATH)
            if os.path.exists(obj_path):
                all_paths.append((ref["name"], obj_path))

                if not os.path.exists(os.path.join(OUT_DIR, ref["name"])):
                    os.makedirs(os.path.join(OUT_DIR, ref["name"]))

            else:
                print("Failed to find", ref["name"], "path:", obj_path)

    with open(PATHS_FILE, "w") as f:
        for path in all_paths:
            f.write("{} {}\n".format(path[0], path[1]))
