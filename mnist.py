from tensorflow.keras import activations
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import wandb
from tensorflow.python.keras.activations import relu
from wandb.keras import WandbCallback

# logging code
run = wandb.init(project="mnist")
config = run.config

config.epochs = 10
config.activation = "softmax"
config.loss = "categorical_crossentropy"
# config.loss_function = "mse"

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
labels = list(range(10))


num_classes = y_train.shape[1]


# create model
model = Sequential()
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dense(num_classes, activation=config.activation))
model.compile(
    # MSE => CCE improves
    loss=config.loss,
    # loss='mse',
    optimizer="adam",
    metrics=["accuracy"],
)

# Fit the model
model.fit(
    X_train,
    y_train,
    epochs=config.epochs,
    validation_data=(X_test, y_test),
    callbacks=[WandbCallback(data_type="image", labels=labels)],
)
