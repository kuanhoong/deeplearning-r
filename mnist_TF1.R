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

library (tensorflow)

# Load the MNIST Data
# MNIST has 55,000 data points of training data (mnist$train)
# 10,000 points of test data (mnist$test), and 
# 5,000 points of validation data (mnist$validation)
datasets <- tf$contrib$learn$datasets

# use one hot encoding to encode 10 digits
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)

# Each image is 28 pixels by 28 pixels
# This array can be flatten into a vector of 28x28 = 784 numbers
# mnist$train$images is a tensor (n-d array with shape (55000L, 784L)
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

# Input the values for x and y
x <- tf$placeholder(tf$float32, shape(NULL, 784L))
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))

# Softmax regressor
# Model y = W*x + b
y <- tf$nn$softmax(tf$matmul(x, W) + b)

# Loss function: Mean Square Error (MSE)
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices=1L))

# Learning rate = 0.5
learning_rate = 0.5

# Gradient Descent Optimizer
optimizer <- tf$train$GradientDescentOptimizer(learning_rate)
train_step <- optimizer$minimize(cross_entropy)

init <- tf$global_variables_initializer()

# Start a new session
sess <- tf$Session()

# Initialize all variables
sess$run(init)

# run 1000 epochs
for (i in 1:1000) {
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  sess$run(train_step, feed_dict = dict(x = batch_xs, y_ = batch_ys))
}

# Model Evaluation
correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

#Display the output
sess$run(accuracy, feed_dict=dict(x = mnist$test$images, y_ = mnist$test$labels))