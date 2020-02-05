# Age and gender recognition (python demo)
Code to run the [Age and Gender Recognition](http://docs.openvinotoolkit.org/latest/_models_intel_age_gender_recognition_retail_0013_description_age_gender_recognition_retail_0013.html) pretrained model from [OpenVINO Model Zoo](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models).

In my case, I use a [Neural Compute Stick 2](https://software.intel.com/neural-compute-stick) (NCS2), so you would need to make some changes to run on CPU, GPU or other devices.

The code is based on the Intel® Edge AI Scholarship Foundation Course from [Udacity](https://www.udacity.com/).

I use Ubuntu Linux 19.10, so it was not tested on Windows nor MacOS.

## How to use
0. I suppose that you have your Intel® Distribution of OpenVINO installed.
1. Download the pretrained model with **./download_model.sh**. Please note that you would need to change the precision (currently FP16, for the NCS2) if you want to run it on CPU, GPU or other devices.
2. On a bash console, import the environment variables: **source /opt/intel/openvino/bin/setupvars.sh**
3. Run: **python3 app.py -i pathToPhotoImage.jpg**
3.1. Optionally, pass -d argument to change the target device (CPU, GPU, etc).
3.2. It should output the Age and Gender "detected" by the model.
