### practice using neural networks for classification using the
### ionosphere data set

library(tidyverse)

data("Ionosphere", package = "mlbench")

### exploration of the data set was performed earlier in the semester
### remove first 2 variables turn the class into a factor
my_ion <- Ionosphere %>% 
  select(-V1, -V2) %>% 
  mutate(Class = factor(Class, levels = c("good", "bad")))

my_ion %>% glimpse()

### let's first fit the model directly with neuralnet
### starting with a single hidden layer with a few hidden units

library(neuralnet)

set.seed(41231)
mod_1layer_small <- neuralnet(Class ~ .,
                              data = my_ion,
                              hidden = 3,
                              err.fct = 'ce',
                              act.fct = 'logistic',
                              linear.output = FALSE,
                              likelihood = TRUE)

plot(mod_1layer_small, rep = "best", show.weights = FALSE)

mod_1layer_small$result.matrix %>% as.data.frame()

### a single hidden layer with more hidden units
set.seed(41231)
mod_1layer <- neuralnet(Class ~ .,
                        data = my_ion,
                        hidden = 15,
                        err.fct = 'ce',
                        act.fct = 'logistic',
                        linear.output = FALSE,
                        likelihood = TRUE)

plot(mod_1layer, rep = "best", show.weights = FALSE)

### use 2 hidden layers with just a small number of hidden units
set.seed(41231)
mod_2layers_small <- neuralnet(Class ~ .,
                               data = my_ion,
                               hidden = c(3, 3),
                               err.fct = 'ce',
                               act.fct = 'logistic',
                               linear.output = FALSE,
                               likelihood = TRUE)

plot(mod_2layers_small, rep = "best", show.weights = TRUE)

### use 2 hidden layers with more hidden units
set.seed(41231)
mod_2layers <- neuralnet(Class ~ .,
                         data = my_ion,
                         hidden = c(15, 7),
                         err.fct = 'ce',
                         act.fct = 'logistic',
                         linear.output = FALSE,
                         likelihood = TRUE)

plot(mod_2layers, rep = "best", show.weights = FALSE, intercept = FALSE)

### use 3 hidden layers with a small number of hidden units
set.seed(41231)
mod_3layers_small <- neuralnet(Class ~ .,
                               data = my_ion,
                               hidden = c(3, 3, 2),
                               err.fct = 'ce',
                               act.fct = 'logistic',
                               linear.output = FALSE,
                               likelihood = TRUE)

plot(mod_3layers_small, rep = "best", show.weights = FALSE, intercept = FALSE)

### use 3 hidden layers with more hidden units
set.seed(41231)
mod_3layers <- neuralnet(Class ~ .,
                         data = my_ion,
                         hidden = c(15, 7, 5),
                         err.fct = 'ce',
                         act.fct = 'logistic',
                         linear.output = FALSE,
                         likelihood = TRUE)

plot(mod_3layers, rep = "best", show.weights = FALSE, intercept = FALSE)

### use AIC / BIC to determine the simplest or best model to use
### which guards against overfitting

mod_1layer_small$result.matrix %>% as.data.frame() %>% 
  tibble::rownames_to_column() %>% 
  tibble::as_tibble() %>% 
  slice(1:5) %>% 
  tidyr::spread(rowname, V1) %>% 
  mutate(layer1 = 3, layer2 = 0, layer3 = 0) %>% 
  bind_rows(mod_1layer$result.matrix %>% as.data.frame() %>% 
              tibble::rownames_to_column() %>% 
              tibble::as_tibble() %>% 
              slice(1:5) %>% 
              tidyr::spread(rowname, V1) %>% 
              mutate(layer1 = 15, layer2 = 0, layer3 = 0)) %>% 
  bind_rows(mod_2layers_small$result.matrix %>% as.data.frame() %>% 
              tibble::rownames_to_column() %>% 
              tibble::as_tibble() %>% 
              slice(1:5) %>% 
              tidyr::spread(rowname, V1) %>% 
              mutate(layer1 = 3, layer2 = 3, layer3 = 0)) %>% 
  bind_rows(mod_2layers$result.matrix %>% as.data.frame() %>% 
              tibble::rownames_to_column() %>% 
              tibble::as_tibble() %>% 
              slice(1:5) %>% 
              tidyr::spread(rowname, V1) %>% 
              mutate(layer1 = 15, layer2 = 7, layer3 = 0)) %>% 
  bind_rows(mod_3layers_small$result.matrix %>% as.data.frame() %>% 
              tibble::rownames_to_column() %>% 
              tibble::as_tibble() %>% 
              slice(1:5) %>% 
              tidyr::spread(rowname, V1) %>% 
              mutate(layer1 = 3, layer2 = 3, layer3 = 2)) %>% 
  bind_rows(mod_3layers$result.matrix %>% as.data.frame() %>% 
              tibble::rownames_to_column() %>% 
              tibble::as_tibble() %>% 
              slice(1:5) %>% 
              tidyr::spread(rowname, V1) %>% 
              mutate(layer1 = 15, layer2 = 7, layer3 = 5)) %>% 
  arrange(aic, bic)

### let's now evaluate the performance using resampling with the caret package
### unfortunately, `caret` does not allow `neuralnet` to be used for classification
### so we will use another simple neural network package, `nnet`

### `nnet` is VERY simple, it only allows for a single hidden layer
### however it provides WEIGHT DECAY to regularize the coefficients

### let's use `caret` to tune the number of hidden units in the single
### hidden layer as well as the regularization strength

library(caret)

### use 5-fold cross validation
ctrl_cv05 <- trainControl(method = "cv", number = 5)

### use the default tuning grid
set.seed(31311)
fit_nnet_default <- train(Class ~ ., data = my_ion,
                          method = "nnet",
                          metric = "Accuracy",
                          preProcess = c("center", "scale"),
                          trControl = ctrl_cv05,
                          trace=FALSE)

fit_nnet_default

### use a more refined grid, `size` is the number of hidden units
### `decay` is the regularization strength
nnet_grid <- expand.grid(size = c(3, 5, 10, 15),
                         decay = exp(seq(-6, 3, length.out = 31)))

set.seed(31311)
fit_nnet_tune <- train(Class ~ ., data = my_ion,
                       method = "nnet",
                       metric = "Accuracy",
                       tuneGrid = nnet_grid,
                       preProcess = c("center", "scale"),
                       trControl = ctrl_cv05,
                       trace=FALSE)

fit_nnet_tune

### plot the resampling results
plot(fit_nnet_tune, xTrans = log)

plot(fit_nnet_tune, top = 20)

### best tuning parameters
fit_nnet_tune$bestTune

### we can access the `nnet` model directly with
fit_nnet_tune$finalModel

### visualiz the network with the NeuralNetTools package

### if interested to learn about the functionality of neuralnettools
### check out it's overview vignette
### http://fawda123.github.io/NeuralNetTools/articles/Overview.html

library(NeuralNetTools)

plotnet(fit_nnet_tune$finalModel)

### which input features matter? garson ONLY works for single hiddenlayer
### and single output model. it tracks all weights, hidden and output layer
### for you! only considers magnitude and not sign
garson(fit_nnet_tune$finalModel)

### more flexible approach is olden's method. allows multiple hidden layers
### and multiple outputs. provides magnitude and sign of the importance
olden(fit_nnet_tune$finalModel)

### alternatively we can use caret's built in variable importance algorithm
### comparing between the previous two, the caret algorithm is the simple garson
### approach 
plot(varImp(fit_nnet_tune))

### look at the confusion matrix
confusionMatrix.train(fit_nnet_tune)

### tune the neural network based on maximizing the area under the
### ROC curve (AUC)

ctrl_cv05_roc <- trainControl(method = "cv", number = 5,
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE,
                              savePredictions = TRUE)

set.seed(31311)
fit_nnet_roc <- train(Class ~ ., data = my_ion,
                      method = "nnet",
                      metric = "ROC",
                      tuneGrid = nnet_grid,
                      preProcess = c("center", "scale"),
                      trControl = ctrl_cv05_roc,
                      trace=FALSE)

fit_nnet_roc$bestTune

confusionMatrix.train(fit_nnet_roc)

plot(fit_nnet_roc, xTrans = log)

plotnet(fit_nnet_roc$finalModel)

### visualize the ROC curve, let's focus 3 specific weight decay values
decay_focus <- nnet_grid %>% distinct(decay) %>% 
  filter(decay %in% c(min(nnet_grid$decay), 
                      fit_nnet_roc$bestTune$decay,
                      max(nnet_grid$decay))) %>% 
  pull(decay)

decay_focus

### use the yardstick package to visualize the roc curve

library(yardstick)

fit_nnet_roc$pred %>% tibble::as_tibble()

### focus on 3 particular weight decay values and 3 sizes of the network
fit_nnet_roc$pred %>% tibble::as_tibble() %>% 
  filter(decay %in% decay_focus,
         size %in% c(3, 5, 15)) %>% 
  group_by(decay, size) %>% 
  roc_curve(obs, good) %>% 
  ggplot(mapping = aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(mapping = aes(group = interaction(size, decay))) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dotted') +
  coord_equal() +
  facet_grid(size ~ decay) +
  theme_bw()

fit_nnet_roc$pred %>% tibble::as_tibble() %>% 
  filter(decay %in% decay_focus,
         size %in% c(3, 5, 15)) %>% 
  group_by(decay, size) %>% 
  roc_curve(obs, good) %>% 
  ggplot(mapping = aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(mapping = aes(group = interaction(size, decay),
                          color = as.factor(decay)),
            size = 1.25) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dotted') +
  coord_equal() +
  facet_wrap(~size) +
  scale_color_viridis_d("decay") +
  theme_bw() +
  theme(legend.position = "top")
