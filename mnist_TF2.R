####################################################
# Deep Learning with R                             #
# Poo Kuan Hoong                                   #
# Malaysia R User Group Meetup                     #
# 13th July 2017                                   #  
# http://www.github.com/kuanhoong/deeplearning-r   #
####################################################

# To install TensorFlow Package
# install_tensorflow()
# devtools::install_github("rstudio/tensorflow")

library(tensorflow)
sess <- tf$InteractiveSession()

# Load the MNIST Data
# MNIST has 55,000 data points of training data (mnist$train)
# 10,000 points of test data (mnist$test), and 
# 5,000 points of validation data (mnist$validation)
input_dataset <- tf$examples$tutorials$mnist$input_data

# use one hot encoding to encode 10 digits
mnist <- input_dataset$read_data_sets("MNIST-data", one_hot = TRUE)

# Each image is 28 pixels by 28 pixels
# This array can be flatten into a vector of 28x28 = 784 numbers
# mnist$train$images is a tensor (n-d array with shape (55000L, 784L)
# Weight Initialization
weight_variable <- function(shape) {
  initial <- tf$truncated_normal(shape, stddev=0.1)
  tf$Variable(initial)
}

bias_variable <- function(shape) {
  initial <- tf$constant(0.1, shape=shape)
  tf$Variable(initial)
}

# Input the values for x and y
x <- tf$placeholder(tf$float32, shape(NULL, 784L))
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))

# Convolution and Pooling
conv2d <- function(x, W) {
  tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')
}

max_pool_2x2 <- function(x) {
  tf$nn$max_pool(
    x, 
    ksize=c(1L, 2L, 2L, 1L),
    strides=c(1L, 2L, 2L, 1L), 
    padding='SAME')
}

# First Convolutional Layer
W_conv1 <- weight_variable(shape(5L, 5L, 1L, 32L))
b_conv1 <- bias_variable(shape(32L))
x_image <- tf$reshape(x, shape(-1L, 28L, 28L, 1L))
h_conv1 <- tf$nn$relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 <- max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 <- weight_variable(shape = shape(5L, 5L, 32L, 64L))
b_conv2 <- bias_variable(shape = shape(64L))
h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 <- max_pool_2x2(h_conv2)

# Densely Connected Layer
W_fc1 <- weight_variable(shape(7L * 7L * 64L, 1024L))
b_fc1 <- bias_variable(shape(1024L))
h_pool2_flat <- tf$reshape(h_pool2, shape(-1L, 7L * 7L * 64L))
h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob <- tf$placeholder(tf$float32)
h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 <- weight_variable(shape(1024L, 10L))
b_fc2 <- bias_variable(shape(10L))

y_conv <- tf$nn$softmax(tf$matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y_conv), reduction_indices=1L))
train_step <- tf$train$AdamOptimizer(1e-4)$minimize(cross_entropy)
correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

init <- tf$global_variables_initializer()

# Initialize all variables
sess$run(init)

# run 20,000 training iterations and may take a while 
# (possibly up to half an hour), depending on your processor.
for (i in 1:20000) {
  batch <- mnist$train$next_batch(50L)
  if (i %% 100 == 0) {
    train_accuracy <- accuracy$eval(feed_dict = dict(
      x = batch[[1]], y_ = batch[[2]], keep_prob = 1.0))
    cat(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
  }
  train_step$run(feed_dict = dict(
    x = batch[[1]], y_ = batch[[2]], keep_prob = 0.5))
}

# Model evaluation
test_accuracy <- accuracy$eval(feed_dict = dict(
  x = mnist$test$images, y_ = mnist$test$labels, keep_prob = 1.0))
cat(sprintf("test accuracy %g", test_accuracy))
