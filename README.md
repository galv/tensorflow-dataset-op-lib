Sample minimal project to build a custom Dataset Op library in tensorflow.

I tested this using the nightly build specified below on Ubuntu 16.04

Usage:

```
pip install tf-nightly-gpu=20180210
./build.sh
conda env create -f env.yml
python test_my_textline_dataset.py
```
