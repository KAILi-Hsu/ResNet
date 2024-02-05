#[Part 1]讀取我們的訓練/測試資料。

library(magrittr)
library(data.table)
library(OpenImageR)

dat <- fread('train.csv', data.table = FALSE)
#data.table = FALSE，這個參數讓資料會是data frame的種類fread()函數拿來讀取csv檔案更快速。

set.seed(123)
all_idx <- 1:nrow(dat)

train_idx <- sample(all_idx, nrow(dat) * 0.7)
valid_idx <- sample(all_idx[!all_idx %in% train_idx], nrow(dat) * 0.15)
test_idx <- all_idx[!all_idx %in% c(train_idx, valid_idx)]

train_dat <- dat[train_idx,]
valid_dat <- dat[valid_idx,]
test_dat <- dat[test_idx,]

#[Part 2]進行u影像資料前處理，將jpeg檔案讀進來並resize成短邊大小為256，語法如下：
image_dir <- 'image/'
processed_dir <- 'processed/'
img_paths <- list.files(image_dir)
pb <- txtProgressBar(max = length(img_paths), style = 3)

for (i in 1:length(img_paths)) {
  
  current_img_path <- paste0(image_dir, img_paths[i])
  current_prc_path <- paste0(processed_dir, gsub('\\.jpg$', '.RData', img_paths[i]))
  
  img <- readImage(current_img_path)
  
  target_size <- c(256, 256)
  targer_fold <- dim(img)[1] / dim(img)[2]
  if (targer_fold > 1) {target_size[1] <- target_size[1] * targer_fold} else {target_size[2] <- target_size[2] / targer_fold}
  resize_img <- resizeImage(img, width = target_size[1], height = target_size[2], method = 'bilinear')
  
  save(resize_img, file = current_prc_path)
  
  setTxtProgressBar(pb, i)
  
}

close(pb)

#[Part 4]為了避免縮放，僅使用圖片中間做為輸入，並將他們合併成一個大陣列：
#train data
processed_dir <- 'processed/'

train_y_array <- model.matrix(~ -1 + factor(train_dat[,'LVD'])) %>% t()
train_img_array <- array(0, dim= c(256, 256,3, nrow(train_dat)))

for (i in 1:nrow(train_dat)){
  
  if (i==1){t0 <- Sys.time()}
  
  current_prc_path <- paste0(processed_dir, train_dat[i,'UID'],'.RData')
  load(current_prc_path)
  
  row_select <- floor((dim(resize_img)[1] - 256)/2) + 1:256
  col_select <- floor((dim(resize_img)[2] - 256)/2) + 1:256
  train_img_array[,,,i] <- resize_img[row_select,col_select,]
  
  if(i %% 100 == 0 ){
    
    batch_time <- as.double(difftime(Sys.time(),t0,units='secs'))
    batch_speed <- i / batch_time
    estimated_mins <- (nrow(train_dat) - i) / batch_speed / 60
    
    message(i, "/", nrow(train_dat), ":Speed ",formatC(batch_speed, 2, format= "f"),
            " samples/sec | Remaining time: ",formatC(estimated_mins, 1, format = "f"),"mins")
  }
}

#valid data
processed_dir <- 'processed/'

valid_y_array <- model.matrix(~ -1 + factor(valid_dat[,'LVD'])) %>% t()
valid_img_array <- array(0, dim = c(256,256,3,nrow(valid_dat)))

for ( i in 1:nrow(valid_dat)){
  
  if( i == 1 ){ t0 <- Sys.time()}
  
  current_prc_path <- paste0(processed_dir, valid_dat[i, 'UID'], '.RData')
  load(current_prc_path)
  
  row_select <- floor((dim(resize_img)[1] - 256)/2) + 1:256
  col_select <- floor((dim(resize_img)[2] - 256)/2) + 1:256
  valid_img_array[,,,i] <- resize_img[row_select, col_select,]
  
  if( i %% 100 == 0){
    
    batch_time <- as.double(difftime(Sys.time(), t0, units = "secs"))
    batch_speed <- i / batch_time
    estimated_mins <- (nrow(valid_dat) - i) / batch_speed / 60
    
    message(i,"/",nrow(valid_dat),": Speed: ",formatC(batch_speed, 2, format = "f"),
            " samples/sec | Remaining time: ",formatC(estimated_mins, 1, format = "f")," mins")
    
  }
  
}

#test data
processed_dir <- 'processed/'

test_y_array <- model.matrix(~ -1 + factor(test_dat[,'LVD'])) %>% t()
test_img_array <- array(0, dim = c(256,256,3,nrow(test_dat)))

for ( i in 1:nrow(test_dat)){
  
  if( i == 1 ){ t0 <- Sys.time()}
  
  current_prc_path <- paste0(processed_dir, test_dat[i, 'UID'], '.RData')
  load(current_prc_path)
  
  row_select <- floor((dim(resize_img)[1] - 256)/2) + 1:256
  col_select <- floor((dim(resize_img)[2] - 256)/2) + 1:256
  test_img_array[,,,i] <- resize_img[row_select, col_select,]
  
  if( i %% 100 == 0){
    
    batch_time <- as.double(difftime(Sys.time(), t0, units = "secs"))
    batch_speed <- i / batch_time
    estimated_mins <- (nrow(test_dat) - i) / batch_speed / 60
    
    message(i,"/",nrow(test_dat),": Speed: ",formatC(batch_speed, 2, format = "f"),
            " samples/sec | Remaining time: ",formatC(estimated_mins, 1, format = "f")," mins")
    
  }
  
}

#[Part5]iterator
my_iterator_core = function(batch_size) {
  
  batch = 0
  batch_per_epoch = length(train_y_array)/batch_size
  
  reset = function() {batch <<- 0}
  
  iter.next = function() {
    batch <<- batch+1
    if (batch > batch_per_epoch) {return(FALSE)} else {return(TRUE)}
  }
  
  value = function() {
    idx = 1:batch_size + (batch - 1) * batch_size
    idx[idx > ncol(train_img_array)] = sample(1:ncol(train_img_array), sum(idx > ncol(train_img_array)))
    data = mx.nd.array(train_img_array[,,,idx, drop=FALSE])
    label = mx.nd.array(train_y_array[,idx, drop=FALSE])
    return(list(data = data, label = label))
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
}

my_iterator_func <- setRefClass("Custom_Iter",
                                fields = c("iter", "batch_size"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods = list(
                                  initialize = function(iter, batch_size){
                                    .self$iter <- my_iterator_core(batch_size = batch_size)
                                    .self
                                  },
                                  value = function(){
                                    .self$iter$value()
                                  },
                                  iter.next = function(){
                                    .self$iter$iter.next()
                                  },
                                  reset = function(){
                                    .self$iter$reset()
                                  },
                                  finalize=function(){
                                  }
                                )
)

my_iter = my_iterator_func(iter = NULL, batch_size = 32)

#[Part6]architecture
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

# #resnet-18
unit <- c(2, 2, 2, 2)
kernel <- c(3, 3)
stride <- c(1, 2, 2, 2)
pad <- c(1, 1)
num_filter <- c(64, 128, 256, 512)
num_filter_num <- c(1, 1)
num_hidden <- 2

#resnet-50
# unit <- c(3, 4, 6, 3)
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

#[Part7]optimizer
my_optimizer = mx.opt.create(name = "adam", learning.rate = 0.001)

#[Part8]開始訓練
my_executor = mx.simple.bind(symbol = m_logloss,
                             data = c(256, 256, 3, 32), label = c(2, 32),
                             ctx = mx.gpu(), grad.req = "write")

mx.set.seed(0)

new_arg = mxnet:::mx.model.init.params(symbol = m_logloss,
                                       input.shape = list(data = c(256, 256, 3, 32), label = c(2, 32)),
                                       output.shape = NULL,
                                       initializer = mxnet:::mx.init.uniform(0.01),
                                       ctx = mx.gpu())

mx.exec.update.arg.arrays(my_executor, new_arg$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(my_executor, new_arg$aux.params, match.name = TRUE)

my_updater = mx.opt.get.updater(optimizer = my_optimizer, weights = my_executor$ref.arg.arrays)

#跑50個epoch
for (i in 1:50) {
  
  my_iter$reset()
  batch_loss = NULL
  
  while (my_iter$iter.next()) {
    
    my_values <- my_iter$value()
    mx.exec.update.arg.arrays(my_executor, arg.arrays = my_values, match.name = TRUE)
    mx.exec.forward(my_executor, is.train = TRUE)
    mx.exec.backward(my_executor)
    update_args = my_updater(weight = my_executor$ref.arg.arrays, grad = my_executor$ref.grad.arrays)
    mx.exec.update.arg.arrays(my_executor, update_args, skip.null = TRUE)
    batch_loss = c(batch_loss, as.array(my_executor$ref.outputs$m_logloss_output))
    
  }
  
  cat(paste0("epoch = ", i, ": m_logloss = ", formatC(mean(batch_loss), format = "f", 4), "\n"))
  
  if (i %% 10 == 0 | i <= 5) {
    cat(paste0("epoch = ", i, ": m_logloss = ", formatC(mean(batch_loss), format = "f", 4), "\n"))
  }
  
}


pred_model <- mxnet:::mx.model.extract.model(symbol = softmax, train.execs = list(my_executor))

#[Part9]訓練完成，將訓練好的model儲存
mx.model.save(model = pred_model, prefix = "~/modelX1/res18_C2", iteration = 49)

#[Part10]將訓練好的模型載入，並進行驗證/測試集預測，並畫出ROC Curve

#載入訓練好的模型
my_model = mx.model.load("~/modelX1/modelX1_res50_S3", iteration = 35)

#對驗證/測試集進行預測
pred_test_v <-predict(model = my_model, X = valid_img_array, ctx = mx.gpu())
pred_test_t <- predict(model = my_model, X = test_img_array, ctx = mx.gpu())

#開始進行繪製前處理與繪製
library(pROC)
valid_y_factor <- as.factor(valid_y_array)
test_y_factor <- as.factor(test_y_array)

# 轉換預測結果為數值向量
pred_test_v_vector <- as.numeric(as.array(pred_test_v))
pred_test_t_vector <- as.numeric(as.array(pred_test_t))
# 創建 ROC 曲線物件
roc_test_valid <- roc(response = valid_y_factor, predictor = pred_test_v_vector)
roc_test_test <- roc(response = test_y_factor, predictor = pred_test_t_vector)

# 繪製 ROC 曲線圖
#plot.roc(roc_test_valid, col = "blue", main = "ROC Curve - CustomResNet18", lwd = 2)
plot.roc(roc_test_valid, col = "blue", main = "ROC Curve - ModifiedResNet50", lwd = 2)
lines.roc(roc_test_test, col = "red", lwd = 2)

# 加上圖例
legend("bottomright", legend = c("Valid Data", "Test Data"), col = c("blue", "red"), lwd = 2)

# 在圖上加入 AUC 值
text(0.7, 0.2, paste("Valid Data AUC =", round(roc_test_valid$auc, 4)), col = "blue")
text(0.7, 0.15, paste("Test Data AUC =", round(roc_test_test$auc, 4)), col = "red")
