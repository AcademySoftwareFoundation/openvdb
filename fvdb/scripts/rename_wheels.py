import sys
import os
import os.path
import glob
import shutil

torch_version = sys.argv[1]
cuda_version = sys.argv[2]

wheels = glob.glob("dist/*.whl")
for wheel in wheels:
    wheel = os.path.basename(wheel)
    filename, ext = os.path.splitext(wheel)
    tags = filename.split("-")
    new_filename = "-".join(
        tags[:-4]
        + [tags[-4] + "+" + "torch" + torch_version + "+" + cuda_version]
        + tags[-3:]
    )
    new_filename += ext
    print(f"Renaming {wheel} -> {new_filename}")
    os.rename(os.path.join("dist", wheel), os.path.join("dist", new_filename))
