from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

class Flixstock_branch_net:
	@staticmethod
	def build_neck_branch(inputs, num_of_classes, final_activation="softmax", channel_dim=-1):
		x = Conv2D(128, (3, 3), padding="same")(inputs)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=channel_dim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)
		x = Flatten()(x)
		x = Dense(256)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Dense(num_of_classes)(x)
		x = Activation(final_activation, name="neck_output")(x)
		return x

	@staticmethod
	def build_sleeves_branch(inputs, num_of_classes, final_activation="softmax", channel_dim=-1):
		x = Conv2D(128, (3, 3), padding="same")(inputs)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=channel_dim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)
		x = Flatten()(x)
		x = Dense(256)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Dense(num_of_classes)(x)
		x = Activation(final_activation, name="sleeves_output")(x)
		return x

	@staticmethod
	def build_pattern_branch(inputs, num_of_classes, final_activation="softmax", channel_dim=-1):
		x = Conv2D(128, (3, 3), padding="same")(inputs)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=channel_dim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Dropout(0.25)(x)
		x = Flatten()(x)
		x = Dense(256)(x)
		x = Activation("relu")(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Dense(num_of_classes)(x)
		x = Activation(final_activation, name="pattern_output")(x)
		return x


