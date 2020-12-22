# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import numpy as np
from tensorflow import keras

from Examples.HouseModel import house_model
from Examples.Mnist import train_mnist

# example 1
print(house_model([7.0]))

# example 2
print(train_mnist())

