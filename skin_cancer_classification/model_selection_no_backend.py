"""
Performs hyperparameter search using Optuna (https://github.com/optuna/optuna-examples/blob/main/keras/keras_simple.py)
Uses datagen.flow_from_dataframe instead of datagen.flow_from_directory.
In this example, we optimize the validation accuracy using
Keras. We optimize hyperparameters such as the filter and kernel size, and layer activation.

References:
https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#pruning
"""

import os
import pickle
import shutil
import sys
import urllib
import warnings
from math import exp

import matplotlib.pyplot as plt
import numpy as np
import optuna

# import argparse
import pandas as pd
import seaborn as sn
import tensorflow as tf
import tensorflow_hub as hub

# To avoid the warning in
# https://github.com/tensorflow/tensorflow/issues/47554
from absl import logging
from keras import regularizers
from keras.applications.resnet import ResNet152, preprocess_input
from keras.backend import clear_session
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from keras.datasets import mnist
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)

# import sklearn.metrics
# from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, f1_score, precision_score, precision_recall_curve
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop

# import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator
from optuna.integration import TFKerasPruningCallback

logging.set_verbosity(logging.ERROR)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# Global variables
ID = 24  # identifier for this simulation - use effnet as backend using val acc instead of AUC (23)
EPOCHS = 10  # maximum number of epochs
IMAGESIZE = (240, 240)  # Define the input shape of the images
INPUTSHAPE = (240, 240, 3)  # NN input
# BEST_MODEL = None # Best NN model
# CURRENT_MODEL = None
VERBOSITY_LEVEL = int(1)  # use 1 to see the progress bar when training and testing

# Important: output folder
OUTPUT_DIR = "../../outputs/optuna_no_backend_outputs/id_" + str(ID) + "/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print("Created folder ", OUTPUT_DIR)

# additional global variables
num_desired_negative_train_examples = 324


def get_data_generators(num_desired_negative_train_examples, batch_size):
    # Define the folders for train, validation, and test data
    train_folder = "../../data_ham1000/HAM10000_images_part_1/"
    validation_folder = "../../data_ham1000/HAM10000_images_part_1/"
    test_folder = "../../data_ham1000/HAM10000_images_part_1/"

    train_csv = "../../data_ham1000/train.csv"
    test_csv = "../../data_ham1000/test.csv"
    validation_csv = "../../data_ham1000/validation.csv"

    # do not remove header
    traindf = pd.read_csv(train_csv, dtype=str)
    testdf = pd.read_csv(test_csv, dtype=str)
    validationdf = pd.read_csv(validation_csv, dtype=str)

    traindf = decrease_num_negatives(traindf, num_desired_negative_train_examples)

    testdf = decrease_num_negatives(testdf, 184)
    validationdf = decrease_num_negatives(validationdf, 76)

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=train_folder,
        x_col="image_name",
        y_col="target",
        target_size=IMAGESIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    # Loading and preprocessing the training, validation, and test data
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validationdf,
        directory=validation_folder,
        x_col="image_name",
        y_col="target",
        target_size=IMAGESIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=test_folder,
        x_col="image_name",
        y_col="target",
        target_size=IMAGESIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )
    return train_generator, validation_generator, test_generator


# not working! CURRENT_MODEL is None
def save_best_model_callback(study, trial):
    global BEST_MODEL, OUTPUT_DIR
    best_model_name = (
        "optuna_best_model"  # do not use .h5 extension to save in modern format
    )
    best_model_name = os.path.join(OUTPUT_DIR, best_model_name)
    if study.best_trial == trial:
        # BEST_MODEL = CURRENT_MODEL
        print("Saving best model ", best_model_name, "...")
        # BEST_MODEL.save(best_model_name)


def simple_NN_objective(trial):  # simple NN
    # Clear clutter from previous Keras session graphs.
    clear_session()

    num_output_neurons = 1

    batch_size = trial.suggest_int("batch_size", 1, 15)
    train_generator, validation_generator, test_generator = get_data_generators(
        num_desired_negative_train_examples, batch_size
    )

    # Define the CNN model
    model = Sequential()

    if True:
        num_conv_layers = 2
        num_conv_filters_L1 = 60
        num_kernel_size_L1 = 8
        num_conv_filters_L2 = 53
        num_kernel_size_L2 = 5
        num_conv_filters_per_layer = np.array(
            [num_conv_filters_L1, num_conv_filters_L2]
        )
        kernel_size_per_layer = np.array([num_kernel_size_L1, num_kernel_size_L2])

    print("kernel_size_per_layer =", kernel_size_per_layer)
    print("num_conv_filters_per_layer =", num_conv_filters_per_layer)
    dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)

    use_batch_normalization = trial.suggest_categorical("batch_nor", [True, False])
    for i in range(num_conv_layers):
        model.add(
            Conv2D(
                # Define the number of filters for the first convolutional layer
                filters=num_conv_filters_per_layer[i],
                # Define the size of the convolutional kernel
                kernel_size=(kernel_size_per_layer[i], kernel_size_per_layer[i]),
                # strides=trial.suggest_categorical("strides", [1, 2]),
                activation=trial.suggest_categorical(
                    "activation", ["relu", "tanh", "elu", "swish"]
                ),
                padding="same",
                input_shape=INPUTSHAPE,
            )
        )
        # first and most important rule is: don't place a BatchNormalization after a Dropout
        # https://stackoverflow.com/questions/59634780/correct-order-for-spatialdropout2d-batchnormalization-and-activation-function
        if use_batch_normalization:
            model.add(BatchNormalization())
        # Define the size of the pooling area for max pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(num_output_neurons, activation="sigmoid"))
    model.summary()

    # We compile our model with a sampled learning rate.
    learning_rate = 1e-3  # trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    # Define the EarlyStopping callback
    if False:
        metric_to_monitor = ("val_accuracy",)
    else:
        metric_to_monitor = ("val_auc",)
    metric_mode = "max"
    early_stopping = EarlyStopping(
        monitor=metric_to_monitor[0],
        patience=3,
        mode=metric_mode,
        restore_best_weights=True,
    )

    # look at https://www.tensorflow.org/guide/keras/serialization_and_saving
    # do not use HDF5 (.h5 extension)
    # best_model_name = 'best_model_' + base_name + '.h5'
    best_model_name = "optuna_best_model_" + str(trial.number)
    best_model_name = os.path.join(OUTPUT_DIR, best_model_name)
    best_model_save = ModelCheckpoint(
        best_model_name,
        save_best_only=True,
        monitor=metric_to_monitor[0],
        mode=metric_mode,
    )

    reduce_lr_loss = ReduceLROnPlateau(
        monitor=metric_to_monitor[0],
        factor=0.5,
        patience=3,
        verbose=VERBOSITY_LEVEL,
        min_delta=1e-4,
        mode=metric_mode,
    )
    # Define Tensorboard as a Keras callback
    tensorboard = TensorBoard(
        log_dir="../outputs/tensorboard_logs",
        histogram_freq=1,
        write_images=True,
    )

    print("")
    print("------------------------------------------------------------")
    print("------------------------------------------------------------")
    print("  Hyperparameters of Optuna trial # ", trial.number)
    print("------------------------------------------------------------")
    print("------------------------------------------------------------")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    model.compile(
        loss="binary_crossentropy",
        # optimizer=RMSprop(learning_rate=learning_rate),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(),
        ],  # always use both metrics, and choose one to guide Optuna
    )

    # Training the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[
            early_stopping,
            best_model_save,
            reduce_lr_loss,
            TFKerasPruningCallback(trial, metric_to_monitor[0]),
            tensorboard,
        ],
    )
    # CURRENT_MODEL = tf.keras.models.clone_model(model)

    # add to history
    history.history["num_desired_train_examples"] = train_generator.samples

    # https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
    pickle_file_path = os.path.join(
        OUTPUT_DIR, "optuna_best_model_" + str(trial.number), "trainHistoryDict.pickle"
    )
    with open(pickle_file_path, "wb") as file_pi:
        pickle.dump(history.history, file_pi)

    # Evaluate the model accuracy on the validation set.
    # score = model.evaluate(x_valid, y_valid, verbose=0)

    if True:
        # train data
        print("Train loss:", history.history["loss"][-1])
        print("Train accuracy:", history.history["accuracy"][-1])
        print("Train AUC:", history.history["auc"][-1])

    if True:  # test data cannot be used in model selection. This is just sanity check
        test_loss, test_accuracy, test_auc = model.evaluate(
            test_generator, verbose=str(VERBOSITY_LEVEL)
        )
        print("Test loss:", test_loss)
        print("Test accuracy:", test_accuracy)
        print("Test AUC:", test_auc)

    # Evaluate the model accuracy on the validation set.
    # val_loss, val_accuracy, val_auc = model.evaluate(validation_generator, verbose=VERBOSITY_LEVEL)
    val_accuracy = history.history["val_accuracy"][-1]
    val_auc = history.history["val_auc"][-1]
    print("Val loss:", history.history["val_loss"][-1])
    print("Val accuracy:", val_accuracy)
    print("Val AUC:", val_auc)

    # Optuna needs to use the same metric for all evaluations (it could be val_accuracy or val_auc but one cannot change it for each trial)
    # return val_accuracy
    return val_auc


def decrease_num_negatives(df, desired_num_negative_examples):
    """
    Create dataframe with desired_num_rows rows from df
    """
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    neg_examples = shuffled_df[shuffled_df["target"] == "0"].copy()
    neg_examples = neg_examples.head(round(desired_num_negative_examples)).copy()

    pos_examples = shuffled_df[shuffled_df["target"] == "1"].copy()
    newdf = pd.concat([neg_examples, pos_examples], ignore_index=True)
    newdf = newdf.sample(frac=1).reset_index(drop=True)  # shuffle again
    return newdf


if __name__ == "__main__":
    print("=====================================")
    print("Model selection")

    # copy script
    copied_script = os.path.join(OUTPUT_DIR, os.path.basename(sys.argv[0]))
    shutil.copy2(sys.argv[0], copied_script)
    print("Just copied current script as file", copied_script)

    # study = optuna.create_study(direction="maximize")
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///../../outputs/optuna_db.sqlite3",  # Specify the storage URL here.
        study_name="ID_" + str(ID),
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    # study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    study.optimize(
        simple_NN_objective, n_trials=40
    )  # , callbacks=[save_best_model_callback]) #, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    trial = study.best_trial
    print("Best trial is #", trial.number)
    print("  Value: {}".format(trial.value))

    print("  Hyperparameters: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open(os.path.join(OUTPUT_DIR, "best_optuna_trial.txt"), "w") as f:
        f.write("Best trial is #" + str(trial.number))
        f.write("\n")
        f.write("  Value: {}".format(trial.value))
        f.write("\n")
        f.write("  Hyperparameters: ")
        f.write("\n")
        for key, value in trial.params.items():
            f.write("    {}: {}".format(key, value))
            f.write("\n")

    pickle_file_path = os.path.join(OUTPUT_DIR, "study.pickle")
    with open(pickle_file_path, "wb") as file_pi:
        pickle.dump(study, file_pi)
    print("Wrote", pickle_file_path)

    # https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_optimization_history.html
    plt.close("all")
    plt.figure()
    # Using optuna.visualization.plot_optimization_history(study) invokes the other Optuna's backend. To use matplotlib, use:
    optuna.visualization.matplotlib.plot_optimization_history(
        study
    )  # optimization history
    # Save the figure to a file (e.g., "optimization_history.png")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "optuna_optimization_history.png"))
    plt.close("all")
    optuna.visualization.matplotlib.plot_intermediate_values(
        study
    )  # Visualize the loss curves of the trials
    plt.savefig(os.path.join(OUTPUT_DIR, "optuna_loss_curves.png"))
    plt.close("all")
    optuna.visualization.matplotlib.plot_contour(study)  # Parameter contour plots
    plt.savefig(os.path.join(OUTPUT_DIR, "optuna_contour_plots.png"))
    plt.close("all")
    optuna.visualization.matplotlib.plot_param_importances(
        study
    )  # parameter importance plot
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "optuna_parameter_importance.png"))
