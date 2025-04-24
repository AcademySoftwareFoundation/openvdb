## Building Tests

1. Activate the `fvdb` environment:
    ```bash
    conda activate fvdb
    ```

2. Build and install fvdb from this README's directory (if not already installed):
    ```bash
    pushd ../.. && python setup.py develop  && popd
    ```

3. Build the tests:
    ```bash
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
    ninja
    ```

4. Run the tests:
    ```bash
    ninja test
    ```
