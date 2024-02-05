library(mxnet)
ResNet <- function(data, unit, kernal, stride, pad, num_filter, num_filter_num, num_hidden){
  
  # data = mx.symbol.Variable(name = 'data')
  # label = mx.symbol.Variable(name = 'label')
  
  bn_data <- mx.symbol.BatchNorm(data = data, eps = "2e-05", name = 'bn_data')
  
  conv0 <- mx.symbol.Convolution(data = data, kernel = c(7, 7), pad = c(3, 3), stride = c(2, 2),num_filter= 64, name = 'conv0')
  bn0 <- mx.symbol.BatchNorm(data = conv0, fix_gamma = FALSE, momentum = 0.9, eps = 1e-5 , name = 'bn0')
  relu0 <- mx.symbol.Activation(data = bn0 , act_type="relu", name = 'relu0')
  
  pooling0 <- mx.symbol.Pooling(data= relu0, pool_type= "max", name ='pooling0',
                                kernel=c(3, 3), stride=c(2, 2), pad =c(1, 1))
  
  #k = model_kind
  
  for(i in 1:4){
    
    if( i == 1 ){
      # data = list(pooling0) ; stage = i ; unit = unit; kernel = kernel; stride = stride[i]; pad = pad
      # num_filter = num_filter[i]; i = i
      
      
      output <- block(data = pooling0 , stage = i , unit = unit, kernel = kernel, stride = stride[i], pad = pad, 
                      num_filter = num_filter[i], i = i)
      
    } else {
      # data = output ; stage = i ; unit = unit; kernel = kernel; stride = stride[i]; pad = pad
      # num_filter = num_filter[i]; i = i
      
      output <- block(data = output, stage = i, unit = unit, kernel = kernel, stride = stride[i], pad = pad, 
                      num_filter = num_filter[i], i = i)
      
    }
    
  }
  
  bn1 <- mx.symbol.BatchNorm(data = output , fix_gamma = FALSE, momentum = 0.9, eps = 1e-5 , name = 'bn1')
  relu1 <- mx.symbol.Activation(data = bn1 , act_type="relu", name = 'relu1')
  pooling1 <- mx.symbol.Pooling(data= relu1, pool_type= "avg", name ='pooling1',
                                kernel=c(7, 7), stride=c(1, 1), pad =c(1, 1))
  
  flatten0 <- mx.symbol.Flatten(data = pooling1, name = 'flatten0')
  fc1 <- mx.symbol.FullyConnected(data = flatten0, num_hidden = num_hidden, name = 'fc1')
  softmax <- mx.symbol.softmax(data = fc1, axis = 1, name = 'softmax')
  
  return(softmax)
}


block <- function(data, stage, unit, kernel, stride, pad, num_filter, i){
  
  plus_num <- c(0, unit)
  
  for(k in 1:unit[i]){
    
    for(j in 1:length(kernel)){
      
      if(j == 1 & k == 1){
        
        data <- conv(data = data, stage = stage, unit = k, num = j, kernel = kernel[j], stride = stride, pad = pad[j],
                     num_filter = num_filter * num_filter_num[j], SC = TRUE)
        
        
        relu_layer <- data[[2]]
        
      } else {
        data <- conv(data = data, stage = stage, unit = k, num = j, kernel = kernel[j], stride = 1, pad = pad[j],
                     num_filter = num_filter * num_filter_num[j], SC = FALSE)
        
      } 
    }
    
    if(k == 1){
      
      sc_layer <- mx.symbol.Convolution(data = relu_layer, no_bias = TRUE, 
                                        name = paste0('stage', stage,'_unit', k, '_sc'),
                                        kernel = c(1, 1), stride = c(stride, stride), pad = c(0, 0),
                                        num_filter = num_filter * num_filter_num[length(num_filter_num)])
      print(mx.symbol.infer.shape(sc_layer, data = c(224, 224, 3, 1))$out.shapes)
      
      plus <- mx.symbol.broadcast_plus(lhs = data[[1]], rhs = sc_layer,
                                       name = paste0('elemwise_add_plus', sum(plus_num[1:i])+k-1))
      print(mx.symbol.infer.shape(plus, data = c(224, 224, 3, 1))$out.shapes)
      
    } else {
      data <- mx.symbol.broadcast_plus(lhs = data[[1]] , rhs = plus , 
                                       name = paste0('elemwise_add_plus', sum(plus_num[1:i])+k-1))
      
      print(mx.symbol.infer.shape(data, data = c(224, 224, 3, 1))$out.shapes)
    }
    
  }
  
  return(data)
}

conv <- function(data, stage, unit, num, kernel, stride, pad, num_filter, SC= SC){
  
  bn <- mx.symbol.BatchNorm(data = data[[1]], fix_gamma = FALSE, eps = "2e-05",
                            name = paste0('stage', stage, '_unit', unit,'_bn', num))
  relu <- mx.symbol.Activation(data = bn, act_type = "relu", 
                               name = paste0('stage', stage,'_unit', unit,'_relu', num))
  conv <- mx.symbol.Convolution(data = relu, no_bias = TRUE,
                                name = paste0('stage', stage,'_unit', unit,'_conv', num),
                                kernel = c(kernel, kernel), stride = c(stride, stride), pad = c(pad, pad), num_filter = num_filter)
  print(mx.symbol.infer.shape(conv, data = c(224, 224, 3, 1))$out.shapes)
  
  if(SC){
    # sc <- mx.symbol.Convolution(data = relu, no_bias = TRUE, 
    #                             name = paste0('stage', stage,'_unit', unit,'_sc'),
    #                             kernel = c(1, 1), stride = c(stride, stride), pad = c(0, 0), num_filter = num_filter)
    # print(mx.symbol.infer.shape(sc, data = c(224, 224, 3, 1))$out.shapes)
    return(c(conv, relu))
  } else {
    return(c(conv))
  }
}

# resnet-18
unit <- c(2, 2, 2, 2)
kernel <- c(3, 3)
stride <- c(1, 2, 2, 2)
pad <- c(1, 1)
num_filter <- c(64, 128, 256, 512)
num_filter_num <- c(1, 1)
num_hidden <- 2

# #resnet-34
# unit <- c(3, 4, 6, 3)
# kernel <- c(3, 3)
# stride <- c(1, 2, 2, 2)
# pad <- c(1, 1)
# num_filter <- c(64, 128, 256, 512)
# num_filter_num <- c(1, 1)
# num_hidden <- 2

#resnet-50
# unit <- c(3, 4, 6, 3)
# kernel <- c(1, 3, 1)
# stride <- c(1, 2, 2, 2)
# pad <- c(0, 1, 0)
# num_filter <- c(64, 128, 256, 512)
# num_filter_num <- c(1, 1, 4)
# num_hidden <- 2

# resnet-101
# unit <- c(3, 4, 23, 3)
# kernel <- c(1, 3, 1)
# stride <- c(1, 2, 2, 2)
# pad <- c(0, 1, 0)
# num_filter <- c(64, 128, 256, 512)
# num_filter_num <- c(1, 1, 4)
# num_hidden <- 9

# resnet-152
# unit <- c(3, 8, 36, 3)
# kernel <- c(1, 3, 1)
# stride <- c(1, 2, 2, 2)
# pad <- c(0, 1, 0)
# num_filter <- c(64, 128, 256, 512)
# num_filter_num <- c(1, 1, 4)
# num_hidden <- 2


data = mx.symbol.Variable(name = 'data')
label = mx.symbol.Variable(name = 'label')

softmax <- ResNet(data, unit, kernal, stride, pad, num_filter, num_filter_num, num_hidden)

#損失函數

eps = 1e-8
m_log = 0 - mx.symbol.mean(mx.symbol.broadcast_mul(mx.symbol.log(softmax + eps), label))
m_logloss = mx.symbol.MakeLoss(m_log, name = 'm_logloss')

# eps <- 1e-8
# ce_loss_pos <- mx.symbol.broadcast_mul(mx.symbol.log(logistic_pred + eps), label)
# ce_loss_neg <- mx.symbol.broadcast_mul(mx.symbol.log(1 - logistic_pred + eps), 1 - label)
# ce_loss_mean <- 0 - mx.symbol.mean(ce_loss_pos + ce_loss_neg)
# loss_symbol <- mx.symbol.MakeLoss(ce_loss_mean, name = 'ce_loss')
