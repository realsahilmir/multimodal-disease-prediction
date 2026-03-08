import zipfile
import json
import tempfile
import os
import shutil


def patch_keras_file(filepath):
    print(f"Patching {filepath}...")

    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist, skipping.")
        return

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Extract to temp dir
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.extractall(temp_dir)

        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.loads(f.read())

        def clean_config(cfg):
            if isinstance(cfg, dict):
                if (
                    cfg.get("class_name") != "InputLayer"
                    and "config" in cfg
                    and isinstance(cfg["config"], dict)
                ):
                    if "batch_input_shape" in cfg["config"]:
                        print(
                            f"Found and removed batch_input_shape in {cfg.get('class_name')}!"
                        )
                        del cfg["config"]["batch_input_shape"]

                if (
                    cfg.get("class_name") == "BatchNormalization"
                    and "config" in cfg
                    and isinstance(cfg["config"], dict)
                ):
                    axis = cfg["config"].get("axis")
                    if isinstance(axis, list) and len(axis) == 1:
                        print("Fixed axis in BatchNormalization!")
                        cfg["config"]["axis"] = axis[0]

                for k, v in cfg.items():
                    clean_config(v)
            elif isinstance(cfg, list):
                for item in cfg:
                    clean_config(item)

        clean_config(config_data)

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(config_data))

        # Archive back
        patched_filepath = filepath.replace(".keras", "_patched.keras")
        with zipfile.ZipFile(patched_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, temp_dir)
                    zf.write(full_path, rel_path)

        print(f"Successfully created patched file: {patched_filepath}")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    patch_keras_file("models/malaria.keras")
    patch_keras_file("models/pneumonia.keras")
