#rewrite model
#對於改寫他人模型訓練，請將全過程程式碼中的Part5-9取代成下列程式碼。

#讀入他人預訓練模型
library(magrittr)
library(mxnet)

res_model <- mx.model.load(prefix = "~/modelX1/resnet-50", iteration = 0)
all_layers <- res_model$symbol$get.internals()
flatten0_output <- which(all_layers$outputs == 'flatten0_output') %>% all_layers$get.output()

fc1 <- mx.symbol.FullyConnected(data = flatten0_output, num_hidden = 2, name = 'fc1')
softmax <- mx.symbol.SoftmaxOutput(data = fc1, name = 'softmax')


#取得模型權重參數
new_arg <- mxnet:::mx.model.init.params(symbol = softmax,
                                        input.shape = list(data = c(224, 224, 3, 32)),
                                        output.shape = NULL,
                                        initializer = mxnet:::mx.init.uniform(0.01),
                                        ctx = mx.cpu())

for (i in 1:length(new_arg$arg.params)) {
  pos <- which(names(res_model$arg.params) == names(new_arg$arg.params)[i])
  if (length(pos) == 1) {
    if (all.equal(dim(res_model$arg.params[[pos]]), dim(new_arg$arg.params[[i]])) == TRUE) {
      new_arg$arg.params[[i]] <- res_model$arg.params[[pos]]
    }
  }
}

#length(pos) == 1要確認上面找的時候只找到唯一一個一樣的名稱的層
#dim()查看維度，確認res_model$arg.params[[pos]]與new_arg$arg.params[[i]]維度相同，all.equal()函數就回回傳TRUE才回接下去把res_model$arg.params傳入new_arg$arg.params


for (i in 1:length(new_arg$aux.params)) {
  pos <- which(names(res_model$aux.params) == names(new_arg$aux.params)[i])
  if (length(pos) == 1) {
    if (all.equal(dim(res_model$aux.params[[pos]]), dim(new_arg$aux.params[[i]])) == TRUE) {
      new_arg$aux.params[[i]] <- res_model$aux.params[[pos]]
    }
  }
}

#開始訓練（若電腦沒有GPU，請將「mx.gpu()」改成「mx.cpu()」）

my.eval.metric.mlogloss <- mx.metric.custom(
  name = "m-logloss", 
  function(real, pred) {
    real1 = as.numeric(as.array(real))
    pred1 = as.numeric(as.array(pred))
    pred1[pred1 <= 1e-6] = 1e-6
    pred1[pred1 >= 1 - 1e-6] = 1 - 1e-6
    return(-mean(real1 * log(pred1), na.rm = TRUE))
  }
)

mx.set.seed(0)

my_model <- mx.model.FeedForward.create(symbol = softmax,
                                        X = train_img_array, y = train_y_array,
                                        optimizer = "sgd", learning.rate = 0.001, momentum = 0.9,
                                        array.batch.size = 32, num.round = 50,
                                        arg.params = new_arg$arg.params, aux.params = new_arg$aux.params,
                                        ctx = mx.gpu(),
                                        eval.metric = my.eval.metric.mlogloss)

#儲存訓練好模型
mx.model.save(my_model, "~/modelX1/res50_R", iteration = 50)
