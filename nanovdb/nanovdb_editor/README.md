##  NanoVDB Editor

WIP

### Prerequisites
- git
- C++ compiler
- CMake
- Python
## Windows
- vcpkg
- Vulkan library `vulkan-1.dll` (usually installed with graphics driver)

### Dependencies
#### Python
```
pip install scikit-build wheel
```
#### Linux
```sh
sudo apt update
sudo apt install libvulkan1
sudo apt install libglfw3 libglfw3-dev
sudo apt install libblosc-dev
sudo apt install libxerces-c-dev
sudo apt install libboost-regex-dev
sudo apt install zlib1g-dev

```
#### MacOS
```sh
# install Vulkan SDK from https://vulkan.lunarg.com/sdk/home#mac
# install homebrew from https://brew.sh
brew install cmake
brew install c-blosc
brew install glfw
brew install xerces-c
brew install boost

```
#### Windows
The dependencies are pulled from vcpkg, to set it up run:
```bat
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
```
Required vcpkg dependencies:
```bat
vcpkg install glfw3
vcpkg install blosc
vcpkg install libe57format
vcpkg install boost-regex
vcpkg install zlib
```

### Assets
Put any data files into `data` folder, which is linked to the location next to the libraries.
Shaders are generated into `shaders/_generated` folder next to the libraries.

### Build and Run
Run build script with `-h` for available build options.

#### Linux
```sh
./build.sh
```

#### Windows
First, rename config file `config` next to the build script to `config.ini` and fill in required env variables:
```
VCPKG_DIR=path/to/vcpkg
MSVS_VERSION="Visual Studio 17 2022"
```
##### Optional variables:
Select profile for Slang compiler (https://github.com/shader-slang/slang/blob/master/source/slang/slang-profile-defs.h):
```
SLANG_PROFILE="sm_5_1"
```
Then, run the build script:
```bat
./build.bat
```

### Editor App
After building, run the editor app:
```sh
./build/Release/pnanovdbeditorapp
```

Run editor without pip installing the package (test local files):
```sh
`python3 pymodule/test.py`
```

### Tests

There are multiple simple apps testing compilation of single shader and showing the editor.

#### C++ Test App
Built with the `./build.{sh|bat}` command, to run:
```sh
./build/Release/pnanovdbeditortestapp
```

#### C Test App
Buld and run with:
```sh
./test_c/build_and_run.sh
```

### Python

#### Windows
Build `./build.bat` with `-p` to build and install the local `nanovdb_editor` package.

#### Linux
Build `./build.sh` with `-p` to build the local `nanovdb_editor` wheel package.

Install built python module from local wheel in conda environment:
```sh
conda env create -f environment.yml
# conda env update -f environment.yml --prune
conda activate nanovdb_editor_env
python3 -m pip install ./pymodule/dist/*.whl --force-reinstall
```

#### Python Test App
Run the test app with:
```sh
python3 test/test_editor.py
```

#### Python Raster Test
```sh
./build.sh -p
python3 raster/test_raster.py
```

### User Params
Viewport shader can have defined struct with user parameters:
```hlsl
struct UserParams
{
    float4 color;
    bool use_color;
    bool3 _pad1;
    int _pad2;
};
```

User params can have defined default values in the json file:
```json
{
    "UserParams": {
        "color": {
            "value": [1.0, 0.0, 1.0, 1.0],
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        }
    }
}
```
Supported types: `bool`, `int`, `uint`, `int64`, `uint64`, `float` and its vectors and 4x4 matrix.
Variables with `_pad` in the name are not shown in the UI.
Those params can be interactively changed with generated UI in the User Params tab.

#### Debugging
Add this line to the python test script to print the PID of the process:
```python
import os
print(os.getpid())
```
##### Linux
1. Build with debug python
```sh
./build.sh -dp
```
1. Run the test script, console output will print the PID of the process
2. Attach to the process with gdb
```sh
gdb -p <PID>
```

##### Windows
1. Build with debug python
```bat
./build.bat -dp
```
1. Run the test script, console output will print the PID of the process
2. Attach to the process in your favorite IDE
