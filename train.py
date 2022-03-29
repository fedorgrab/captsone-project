from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def plot_fitting_history(values, colors, labels, epoch_num, title):
    for i in range(len(labels)):
        plt.plot(
            list(range(1, epoch_num + 1)),
            values[i],
            marker=".",
            c=colors[i],
            label=labels[i],
        )

    plt.title(title)
    if epoch_num <= 50:
        plt.xticks(list(range(0, epoch_num + 1, 2)))
    else:
        plt.xticks(list(range(0, epoch_num + 1, 20)))
    plt.legend()
    plt.savefig("training_history.png")


embedding_layer = layers.Embedding(1000, 5)
(train_data, test_data), info = tfds.load(
    "imdb_reviews/subwords8k",
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True,
    as_supervised=True,
)
encoder = info.features["text"].encoder
padded_shapes = ([None], ())
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes=padded_shapes)
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes=padded_shapes)

embedding_dim = 16

model = keras.Sequential(
    [
        layers.Embedding(encoder.vocab_size, embedding_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ]
)


if __name__ == "__main__":
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_batches, epochs=12, validation_data=test_batches, validation_steps=20
    )

    train_accuracy = history.history["accuracy"]
    valid_accuracy = history.history["val_accuracy"]
    plot_fitting_history(
        [train_accuracy, valid_accuracy],
        ["orange", "blue"],
        ["train accuracy", "valid accuracy"],
        epoch_num=12,
        title="Train and validation accuracy",
    )
    evaluation = model.evaluate(test_batches)
    with open("metrics.txt", "w") as outfile:
        outfile.write("Loss: " + str(evaluation[0]) + "\n")
        outfile.write("Accuracy: " + str(evaluation[1]) + "\n")
