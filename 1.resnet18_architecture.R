# Model Architecture
# 224×224

data <- mx.symbol.Variable(name = 'data')
bn_data <- mx.symbol.BatchNorm(data = data, eps = "2e-05", name = 'bn_data')

# 112×112

conv0 <- mx.symbol.Convolution(data = bn_data, no_bias = TRUE, name = 'conv0',
                               kernel = c(7, 7), pad = c(3, 3), stride = c(2, 2), num_filter = 64)
bn0 <- mx.symbol.BatchNorm(data = conv0, fix_gamma = FALSE, eps = "2e-05", name = 'bn0')
relu0 <- mx.symbol.Activation(data = bn0, act_type = "relu", name = 'relu0')

# 56×56

# stage1_unit1

pooling0 <- mx.symbol.Pooling(data = relu0, pool_type = "max", name = 'pooling0',
                              kernel = c(3, 3), pad = c(1, 1), stride = c(2, 2))
stage1_unit1_bn1 <- mx.symbol.BatchNorm(data = pooling0, fix_gamma = FALSE, eps = "2e-05", name = 'stage1_unit1_bn1')
stage1_unit1_relu1 <- mx.symbol.Activation(data = stage1_unit1_bn1, act_type = "relu", name = 'stage1_unit1_relu1')
stage1_unit1_conv1 <- mx.symbol.Convolution(data = stage1_unit1_relu1, no_bias = TRUE, name = 'stage1_unit1_conv1',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 64)
stage1_unit1_bn2 <- mx.symbol.BatchNorm(data = stage1_unit1_conv1, fix_gamma = FALSE, eps = "2e-05", name = 'stage1_unit1_bn2')
stage1_unit1_relu2 <- mx.symbol.Activation(data = stage1_unit1_bn2, act_type = "relu", name = 'stage1_unit1_relu2')
stage1_unit1_conv2 <- mx.symbol.Convolution(data = stage1_unit1_relu2, no_bias = TRUE, name = 'stage1_unit1_conv2',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 64)

stage1_unit1_sc <- mx.symbol.Convolution(data = stage1_unit1_relu1, no_bias = TRUE, name = 'stage1_unit1_sc',
                                         kernel = c(1, 1), pad = c(0, 0), stride = c(1, 1), num_filter = 64)

elemwise_add_plus0 <- mx.symbol.broadcast_plus(lhs = stage1_unit1_conv2, rhs = stage1_unit1_sc, name = 'elemwise_add_plus0')

# stage1_unit2

stage1_unit2_bn1 <- mx.symbol.BatchNorm(data = elemwise_add_plus0, fix_gamma = FALSE, eps = "2e-05", name = 'stage1_unit2_bn1')
stage1_unit2_relu1 <- mx.symbol.Activation(data = stage1_unit2_bn1, act_type = "relu", name = 'stage1_unit2_relu1')
stage1_unit2_conv1 <- mx.symbol.Convolution(data = stage1_unit2_relu1, no_bias = TRUE, name = 'stage1_unit2_conv1',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 64)
stage1_unit2_bn2 <- mx.symbol.BatchNorm(data = stage1_unit2_conv1, fix_gamma = FALSE, eps = "2e-05", name = 'stage1_unit2_bn2')
stage1_unit2_relu2 <- mx.symbol.Activation(data = stage1_unit2_bn2, act_type = "relu", name = 'stage1_unit2_relu2')
stage1_unit2_conv2 <- mx.symbol.Convolution(data = stage1_unit2_relu2, no_bias = TRUE, name = 'stage1_unit2_conv2',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 64)

elemwise_add_plus1 <- mx.symbol.broadcast_plus(lhs = stage1_unit2_conv2, rhs = elemwise_add_plus0, name = 'elemwise_add_plus1')

# 28×28

# stage2_unit1

stage2_unit1_bn1 <- mx.symbol.BatchNorm(data = elemwise_add_plus1, fix_gamma = FALSE, eps = "2e-05", name = 'stage2_unit1_bn1')
stage2_unit1_relu1 <- mx.symbol.Activation(data = stage2_unit1_bn1, act_type = "relu", name = 'stage2_unit1_relu1')
stage2_unit1_conv1 <- mx.symbol.Convolution(data = stage2_unit1_relu1, no_bias = TRUE, name = 'stage2_unit1_conv1',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(2, 2), num_filter = 128)
stage2_unit1_bn2 <- mx.symbol.BatchNorm(data = stage2_unit1_conv1, fix_gamma = FALSE, eps = "2e-05", name = 'stage2_unit1_bn2')
stage2_unit1_relu2 <- mx.symbol.Activation(data = stage2_unit1_bn2, act_type = "relu", name = 'stage2_unit1_relu2')
stage2_unit1_conv2 <- mx.symbol.Convolution(data = stage2_unit1_relu2, no_bias = TRUE, name = 'stage2_unit1_conv2',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 128)

stage2_unit1_sc <- mx.symbol.Convolution(data = stage2_unit1_relu1, no_bias = TRUE, name = 'stage2_unit1_sc',
                                         kernel = c(1, 1), pad = c(0, 0), stride = c(2, 2), num_filter = 128)

elemwise_add_plus2 <- mx.symbol.broadcast_plus(lhs = stage2_unit1_conv2, rhs = stage2_unit1_sc, name = 'elemwise_add_plus2')

# stage2_unit2

stage2_unit2_bn1 <- mx.symbol.BatchNorm(data = elemwise_add_plus2, fix_gamma = FALSE, eps = "2e-05", name = 'stage2_unit2_bn1')
stage2_unit2_relu1 <- mx.symbol.Activation(data = stage2_unit2_bn1, act_type = "relu", name = 'stage2_unit2_relu1')
stage2_unit2_conv1 <- mx.symbol.Convolution(data = stage2_unit2_relu1, no_bias = TRUE, name = 'stage2_unit2_conv1',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 128)
stage2_unit2_bn2 <- mx.symbol.BatchNorm(data = stage2_unit2_conv1, fix_gamma = FALSE, eps = "2e-05", name = 'stage2_unit2_bn2')
stage2_unit2_relu2 <- mx.symbol.Activation(data = stage2_unit2_bn2, act_type = "relu", name = 'stage2_unit2_relu2')
stage2_unit2_conv2 <- mx.symbol.Convolution(data = stage2_unit2_relu2, no_bias = TRUE, name = 'stage2_unit2_conv2',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 128)

elemwise_add_plus3 <- mx.symbol.broadcast_plus(lhs = stage2_unit2_conv2, rhs = elemwise_add_plus2, name = 'elemwise_add_plus3')

# 14×14

# stage3_unit1

stage3_unit1_bn1 <- mx.symbol.BatchNorm(data = elemwise_add_plus3, fix_gamma = FALSE, eps = "2e-05", name = 'stage3_unit1_bn1')
stage3_unit1_relu1 <- mx.symbol.Activation(data = stage3_unit1_bn1, act_type = "relu", name = 'stage3_unit1_relu1')
stage3_unit1_conv1 <- mx.symbol.Convolution(data = stage3_unit1_relu1, no_bias = TRUE, name = 'stage3_unit1_conv1',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(2, 2), num_filter = 256)
stage3_unit1_bn2 <- mx.symbol.BatchNorm(data = stage3_unit1_conv1, fix_gamma = FALSE, eps = "2e-05", name = 'stage3_unit1_bn2')
stage3_unit1_relu2 <- mx.symbol.Activation(data = stage3_unit1_bn2, act_type = "relu", name = 'stage3_unit1_relu2')
stage3_unit1_conv2 <- mx.symbol.Convolution(data = stage3_unit1_relu2, no_bias = TRUE, name = 'stage3_unit1_conv2',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 256)

stage3_unit1_sc <- mx.symbol.Convolution(data = stage3_unit1_relu1, no_bias = TRUE, name = 'stage3_unit1_sc',
                                         kernel = c(1, 1), pad = c(0, 0), stride = c(2, 2), num_filter = 256)

elemwise_add_plus4 <- mx.symbol.broadcast_plus(lhs = stage3_unit1_conv2, rhs = stage3_unit1_sc, name = 'elemwise_add_plus4')

# stage3_unit2

stage3_unit2_bn1 <- mx.symbol.BatchNorm(data = elemwise_add_plus4, fix_gamma = FALSE, eps = "2e-05", name = 'stage3_unit2_bn1')
stage3_unit2_relu1 <- mx.symbol.Activation(data = stage3_unit2_bn1, act_type = "relu", name = 'stage3_unit2_relu1')
stage3_unit2_conv1 <- mx.symbol.Convolution(data = stage3_unit2_relu1, no_bias = TRUE, name = 'stage3_unit2_conv1',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 256)
stage3_unit2_bn2 <- mx.symbol.BatchNorm(data = stage3_unit2_conv1, fix_gamma = FALSE, eps = "2e-05", name = 'stage3_unit2_bn2')
stage3_unit2_relu2 <- mx.symbol.Activation(data = stage3_unit2_bn2, act_type = "relu", name = 'stage3_unit2_relu2')
stage3_unit2_conv2 <- mx.symbol.Convolution(data = stage3_unit2_relu2, no_bias = TRUE, name = 'stage3_unit2_conv2',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 256)

elemwise_add_plus5 <- mx.symbol.broadcast_plus(lhs = stage3_unit2_conv2, rhs = elemwise_add_plus4, name = 'elemwise_add_plus5')

# 7×7

# stage4_unit1

stage4_unit1_bn1 <- mx.symbol.BatchNorm(data = elemwise_add_plus5, fix_gamma = FALSE, eps = "2e-05", name = 'stage4_unit1_bn1')
stage4_unit1_relu1 <- mx.symbol.Activation(data = stage4_unit1_bn1, act_type = "relu", name = 'stage4_unit1_relu1')
stage4_unit1_conv1 <- mx.symbol.Convolution(data = stage4_unit1_relu1, no_bias = TRUE, name = 'stage4_unit1_conv1',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(2, 2), num_filter = 512)
stage4_unit1_bn2 <- mx.symbol.BatchNorm(data = stage4_unit1_conv1, fix_gamma = FALSE, eps = "2e-05", name = 'stage4_unit1_bn2')
stage4_unit1_relu2 <- mx.symbol.Activation(data = stage4_unit1_bn2, act_type = "relu", name = 'stage4_unit1_relu2')
stage4_unit1_conv2 <- mx.symbol.Convolution(data = stage4_unit1_relu2, no_bias = TRUE, name = 'stage4_unit1_conv2',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 512)

stage4_unit1_sc <- mx.symbol.Convolution(data = stage4_unit1_relu1, no_bias = TRUE, name = 'stage4_unit1_sc',
                                         kernel = c(1, 1), pad = c(0, 0), stride = c(2, 2), num_filter = 512)

elemwise_add_plus6 <- mx.symbol.broadcast_plus(lhs = stage4_unit1_conv2, rhs = stage4_unit1_sc, name = 'elemwise_add_plus6')

# stage4_unit2

stage4_unit2_bn1 <- mx.symbol.BatchNorm(data = elemwise_add_plus6, fix_gamma = FALSE, eps = "2e-05", name = 'stage4_unit2_bn1')
stage4_unit2_relu1 <- mx.symbol.Activation(data = stage4_unit2_bn1, act_type = "relu", name = 'stage4_unit2_relu1')
stage4_unit2_conv1 <- mx.symbol.Convolution(data = stage4_unit2_relu1, no_bias = TRUE, name = 'stage4_unit2_conv1',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 512)
stage4_unit2_bn2 <- mx.symbol.BatchNorm(data = stage4_unit2_conv1, fix_gamma = FALSE, eps = "2e-05", name = 'stage4_unit2_bn2')
stage4_unit2_relu2 <- mx.symbol.Activation(data = stage4_unit2_bn2, act_type = "relu", name = 'stage4_unit2_relu2')
stage4_unit2_conv2 <- mx.symbol.Convolution(data = stage4_unit2_relu2, no_bias = TRUE, name = 'stage4_unit2_conv2',
                                            kernel = c(3, 3), pad = c(1, 1), stride = c(1, 1), num_filter = 512)

elemwise_add_plus7 <- mx.symbol.broadcast_plus(lhs = stage4_unit2_conv2, rhs = elemwise_add_plus6, name = 'elemwise_add_plus7')

# Final

bn1 <- mx.symbol.BatchNorm(data = elemwise_add_plus7, fix_gamma = FALSE, eps = "2e-05", name = 'bn1')
relu1 <- mx.symbol.Activation(data = bn1, act_type = "relu", name = 'relu1')
pool1 <- mx.symbol.Pooling(data = relu1, pool_type = "avg", name = 'pool1',
                           kernel = c(7, 7), pad = c(0, 0), stride = c(7, 7))
flatten0 <- mx.symbol.Flatten(data = pool1, name = 'flatten0')
fc1 <- mx.symbol.FullyConnected(data = flatten0, num_hidden = 1000, name = 'fc1')
softmax <- mx.symbol.softmax(data = fc1, axis = 1, name = 'softmax')
