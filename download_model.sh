#!/bin/bash
#
# Downloads the pretrained model with the Model Downloader
# Caution: I am using NCS2 (MYRIAD device), so I only need FP16 precision. Change it if you need

MODEL_NAME=age-gender-recognition-retail-0013

#Downloading FP16 model (only for MYRIAD device)
PRECISIONS=FP16

DOWNLOAD_DIR=intel/${MODEL_NAME}/${PRECISIONS}

if [[ "$OSTYPE" != "linux-gnu" ]]; then
	echo "Not using Linux. Use at your own risk"
fi

if [[ -f $DOWNLOAD_DIR/${MODEL_NAME}.xml && -f $DOWNLOAD_DIR/${MODEL_NAME}.bin ]]; then
	echo "Model already downloaded. Skipping"
	exit 0
fi

if [ -z "$INTEL_OPENVINO_DIR" ]; then
	source /opt/intel/openvino/bin/setupvars.sh
fi

$INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py -o . --precisions $PRECISIONS --name $MODEL_NAME
