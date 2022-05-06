import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape, InputLayer, Conv2DTranspose
import os


class AutoEncoder(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, latent_dim, input_shape, nu_of_filters=16, filter_size1=(2, 2),
                 filter_size2=(3, 3), pooling_size1=(2, 2), pooling_size2=(2, 2)):
        super(AutoEncoder, self).__init__()
        self.shape = input_shape
        # encoder
        self.encoder = []
        self.encoder.append(InputLayer(input_shape=(input_shape[0], input_shape[1], 1)))
        self.encoder.append(Conv2D(nu_of_filters, activation="relu", padding="same", strides=pooling_size1, kernel_size=filter_size1[0]))
        self.encoder.append(Conv2D(nu_of_filters, activation="relu", padding="same", strides=pooling_size1, kernel_size=filter_size1[0]))
        # self.encoder.append(MaxPooling2D(pooling_size1, padding="same"))
        self.encoder.append(Flatten())
        # self.encoder.append(Dense(latent_dim * 8, activation='relu'))
        # self.encoder.append(Dense(latent_dim * 4, activation='relu'))
        self.encoder.append(Dense(latent_dim * 2, activation='relu'))
        self.encoder.append(Dense(latent_dim, activation='sigmoid'))

        # decoder
        self.decoder = []
        self.decoder.append(Dense(latent_dim * 2, activation='relu'))
        self.decoder.append(Dense(7 * 7 * nu_of_filters, activation='relu'))
        # self.decoder.append(Dense(latent_dim * 4, activation='relu'))
        # self.decoder.append(Dense(latent_dim * 8, activation='relu'))
        # self.decoder.append(Dense(14 * 14 * nu_of_filters, activation='relu'))
        self.decoder.append(Reshape((7, 7, nu_of_filters)))
        self.decoder.append(Conv2DTranspose(nu_of_filters, filter_size1, strides=2, activation="relu", padding="same"))
        self.decoder.append(Conv2DTranspose(nu_of_filters, filter_size1, strides=2, activation="relu", padding="same"))
        self.decoder.append(Conv2DTranspose(1, filter_size1, activation="sigmoid", padding="same", strides=1))

        self.epochs = 3000
        self.current_epoch = 0
        self.is_saved = False

    def encode(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, (-1, self.shape[0], self.shape[1], 1))
        for layer in self.encoder:
            state = layer(state)
        return state

    def decode(self, encoded):
        for layer in self.decoder:
            encoded = layer(encoded)
        return encoded

    def feed_forward(self, state):
        return self.decode(self.encode(state))

    def needs_to_learn(self):
        return self.current_epoch < self.epochs and not self.is_saved

    def learn(self, states_raw):
        states_raw = tf.convert_to_tensor(states_raw, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            # calculating auto encoder loss
            auto_states_guess = self.feed_forward(states_raw)
            states_raw = tf.reshape(states_raw, (-1, self.shape[0], self.shape[1], 1))
            auto_loss = tf.losses.MSE(states_raw, auto_states_guess)

        auto_gradient = tape.gradient(auto_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(auto_gradient, self.trainable_variables))
        self.current_epoch += 1

    def save_agent(self, path):
        if not self.needs_to_learn():
            return
        self.save_weights(path)
        print("Auto encoder saved")
        self.is_saved = True
