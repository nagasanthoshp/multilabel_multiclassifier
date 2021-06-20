import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import numpy as np

from flixstock_classifier import Flixstock_branch_net


class MultiLabelMutliClassifierNet:
    def __init__(self, 
                 input_shape, 
                 num_neck_classes, 
                 num_sleeves_classes, 
                 num_pattern_classes,
                 finalAct="softmax"
                ):
        
        mnv2 = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                                              alpha=0.5, 
                                                              include_top=False, 
                                                              weights='imagenet')
        
        finetune = True
        if (finetune):
            for layer in mnv2.layers[:-3]:
                layer.trainable = False

        mobnet_interest_layer = mnv2.layers[-4].output

        # constructs sub branches 
        chanDim = -1
        neck_branch = Flixstock_branch_net.build_neck_branch(mobnet_interest_layer,
            num_neck_classes, final_activation=finalAct, channel_dim=chanDim)

        sleeves_branch = Flixstock_branch_net.build_sleeves_branch(mobnet_interest_layer,
            num_sleeves_classes, final_activation=finalAct, channel_dim=chanDim)

        pattern_branch = Flixstock_branch_net.build_pattern_branch(mobnet_interest_layer,
            num_pattern_classes, final_activation=finalAct, channel_dim=chanDim)
        
        self.model = Model(inputs = mnv2.input, 
                                 outputs = [neck_branch, sleeves_branch, pattern_branch], 
                                 name='flixstock_classifier')
    
    def compile(self,
                losses,
                lossWeights,
                opt='Adam',
                metrics=['accuracy'],
                ):

        self.model.compile(optimizer=opt,
                           loss=losses,
                           metrics=metrics,
                          loss_weights=lossWeights)
        self._is_model_compiled = True
    
    def __is_compile_called(self):

        assert hasattr(self, '_is_model_compiled'), (
            'You must compile your model before calling train/evaluate'
        )
    
    def load_weights(self,
                    checkpoint_path):

        self.model.load_weights(checkpoint_path)
        self._model_trained = True

    def train(self,
              train_x,
              train_y,
              val_x,
              val_y,
              epochs,
              batch_size=32,
              callbacks=[]):
        
        self.__is_compile_called()
        
        self.training_history = self.model.fit(x=train_x,
                                    y=train_y,
                                    validation_data=(val_x, val_y),
                                    epochs=epochs,
                                    verbose=1,
                                    batch_size=batch_size,
                                    callbacks=callbacks)
        self._model_trained = True


    def evaluate(self, test_data_generator, batch_size=32):
        
        self.__is_compile_called()

        assert hasattr(self, '_model_trained'), (
            'model is not trained or load checkpoint , either `train` the model or load the `weights`'
        )

        self.model.evaluate(test_data_generator, steps=test_data_generator.samples // batch_size)

    def __check_na_and_orig(self, label):
        if label==0:
            return 'NA'
        else:
            return label - 1

    def __get_orig_label(self, neck, sleeve, pattern):
        neck, sleeve, pattern = self.__check_na_and_orig(np.argmax(neck)), self.__check_na_and_orig(np.argmax(sleeve)), self.__check_na_and_orig(np.argmax(pattern))
        return neck, sleeve, pattern

    def inference(self, img_path):
        img = cv2.imread(img_path)
        try:
            print('inference image size = {}'.format(img.shape))
        except:
            raise "Error with Image"
        img = img / 255.0 
        img = np.expand_dims(img, 0)
        assert len(img.shape) == 4
        neck_prob, sleeve_prob, pattern_prob = self.model.predict(img)
        neck_orig_label, sleeve_orig_label, pattern_orig_label = self.__get_orig_label(neck_prob, sleeve_prob, pattern_prob)
        return neck_orig_label, sleeve_orig_label, pattern_orig_label