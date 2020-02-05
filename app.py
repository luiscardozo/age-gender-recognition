import argparse
import cv2
import numpy as np
from inference import Network

DEFAULT_MODEL = './intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml'
CPU_EXTENSION = '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Detect gender and age")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'MYRIAD' (e.g.: CPU, GPU)"
    
    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    optional.add_argument("-m", help=m_desc, default=DEFAULT_MODEL)
    optional.add_argument("-d", help=d_desc, default='MYRIAD')
    args = parser.parse_args()

    return args

def preprocess(img, height, width):
    '''
    Prepare the image for the requirements of the model
    '''
    new_frame = cv2.resize(img, (width, height))
    new_frame = new_frame.transpose((2,0,1))        # requires CxHxW, was HxWxC
    new_frame = new_frame.reshape(1, *new_frame.shape)  #1xCxHxW (*new_frame.shape == 3xHxW)
    return new_frame


def infer_on_photo(args):
    #Input: [1x3x62x62] - [1xCxHxW]
    #Outputs:
        # name: "age_conv3", shape: [1, 1, 1, 1] - Estimated age divided by 100.
        # name: "prob", shape: [1, 2, 1, 1] - Softmax output across 2 type classes [female, male]

    engine = Network()
    engine.load_model(args.m, args.d, CPU_EXTENSION)
    image = cv2.imread(args.i)
    
    net_shape = engine.get_input_shape()    #[1, 3, 62, 62]
    image = preprocess(image, net_shape[2], net_shape[3])   #H, W

    engine.async_inference(image)

    if(engine.wait() == 0):
        output = engine.extract_outputs()
        #print(output)
        age = int(output['age_conv3'][0][0][0][0]*100)
        genderM = output['prob'][0][1][0][0]
        gender = "Masculine" if genderM > 0.5 else "Femenine"
        print("Age: %d, gender: %s" % (age, gender))


def main():
    args = get_args()
    #infer_on_video(args)
    infer_on_photo(args)


if __name__ == "__main__":
    main()
