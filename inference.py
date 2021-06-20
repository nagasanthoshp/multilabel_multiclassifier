import cv2
import numpy as  np
import sys
import os
from tensorflow.keras.optimizers import Adam
import argparse

from multilabel_multiclass_classifier_net import MultiLabelMutliClassifierNet

def main(args):
    image_path = args.test_image_path
    checkpoint_path = args.model_checkpoint

    assert os.path.exists(image_path)
    assert os.path.exists(checkpoint_path)

    input_shape = (300, 225, 3)
    neck_classes = 8
    sleeves_classes = 5
    pattern_classes = 11
    net = MultiLabelMutliClassifierNet(input_shape, 
                    num_neck_classes=neck_classes, 
                    num_sleeves_classes=sleeves_classes, 
                    num_pattern_classes=pattern_classes)

    # Define loss and loss weight (if any)
    losses = {
        "pattern_output": "categorical_crossentropy",
        "sleeves_output": "categorical_crossentropy",
        "neck_output": "categorical_crossentropy"
    }
    lossWeights = {"neck_output": 1.0, "sleeves_output": 1.0, "pattern_output": 1.0}

    EPOCHS = 50
    INIT_LR = 1e-3

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    net.compile(losses=losses, lossWeights=lossWeights, opt=opt)

    # checkpoint_path = './checkpoints_20-06/classification_epoch_11_valloss_2.96.h5'
    net.load_weights(checkpoint_path)


    neck, sleeves, pattern = net.inference(image_path)
    # neck, sleeves, pattern = net.model.predict()
    print(neck, sleeves, pattern)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--test_image_path", required=True, help="path to test image", type=str)
    parser.add_argument("-c", "--model_checkpoint", required=True, help="path to checkpoint", type=str)
    args = parser.parse_args()
    main(args)