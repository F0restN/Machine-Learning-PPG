### regression and classification example with random forests and
### boosted trees using caret

### packages needed to run this script
### tidyverse (you should have)
### modeldata (don't need to load in this session but we use it)
### caret
### yardstick (for visualizing ROC curves)
###
### caret you tell you what other packages you need for fitting models
### please check the console accordingly

library(tidyverse)

library(caret)

### load in the data from the modeldata package

data("concrete", package = "modeldata")

concrete %>% glimpse()

### check the names of the variables
concrete %>% names()

### as described earlier in the semester in the complete
### concrete regression demo with tidymodels, simplify this problem slightly
### by grouping the replications and thus focus just on the 
### AVERAGE strength
my_concrete <- concrete %>% 
  group_by(across(cement:age)) %>% 
  summarise(compressive_strength = mean(compressive_strength),
            .groups = 'drop')

### check the variable names again
my_concrete %>% glimpse()

### fit a random forest model without tuning mtry, we will just fit the 
### model with the whole data set, we will discuss evaluating the model
### performance using the OUT OF BAG (OOB) error first then we will
### tune mtry with resampling
ctrl_nocv <- trainControl(method = "none")

### use a value of mtry=2
set.seed(12341)
fit_rf_nocv_2 <- train(compressive_strength ~ .,
                       data = my_concrete,
                       method = "rf",
                       metric = "RMSE",
                       trControl = ctrl_nocv,
                       tuneGrid = expand.grid(mtry = 2),
                       importance = TRUE)

fit_rf_nocv_2

fit_rf_nocv_2$finalModel

### plot the OOB error as a function of the number
### of trees
tibble::tibble(
  n_tree = 1:length(fit_rf_nocv_2$finalModel$mse),
  oob_mse = fit_rf_nocv_2$finalModel$mse
) %>% 
  mutate(mtry = 2) %>% 
  ggplot(mapping = aes(x = n_tree, y = oob_mse)) +
  geom_line(size = 1.15) +
  facet_grid(. ~ mtry, labeller = "label_both") +
  labs(x = "number of trees", y = "OOB error") +
  theme_bw()

### use a value of mtry=4
set.seed(12341)
fit_rf_nocv_4 <- train(compressive_strength ~ .,
                       data = my_concrete,
                       method = "rf",
                       metric = "RMSE",
                       trControl = ctrl_nocv,
                       tuneGrid = expand.grid(mtry = 4),
                       importance = TRUE)

fit_rf_nocv_4$finalModel

### try again with the max number of inputs: 8
set.seed(12341)
fit_rf_nocv_8 <- train(compressive_strength ~ .,
                       data = my_concrete,
                       method = "rf",
                       metric = "RMSE",
                       trControl = ctrl_nocv,
                       tuneGrid = expand.grid(mtry = 8),
                       importance = TRUE)

fit_rf_nocv_8$finalModel

### compare the 3 specific values based on the OOB error
tibble::tibble(
  n_tree = seq_along(fit_rf_nocv_2$finalModel$mse),
  oob_mse = fit_rf_nocv_2$finalModel$mse
) %>% 
  mutate(mtry = 2) %>% 
  bind_rows(tibble::tibble(n_tree = seq_along(fit_rf_nocv_4$finalModel$mse),
                           oob_mse = fit_rf_nocv_4$finalModel$mse) %>% 
              mutate(mtry = 4)) %>% 
  bind_rows(tibble::tibble(n_tree = seq_along(fit_rf_nocv_8$finalModel$mse),
                           oob_mse = fit_rf_nocv_8$finalModel$mse) %>% 
              mutate(mtry = 8)) %>% 
  ggplot(mapping = aes(x = n_tree,
                       y = oob_mse)) +
  geom_line(size = 1.15,
            mapping = aes(color = as.factor(mtry),
                          linetype = as.factor(mtry))) +
  ggthemes::scale_color_colorblind("mtry") +
  scale_linetype_discrete("mtry") +
  labs(x = "number of trees",
       y = "OOB error") +
  theme_bw() +
  theme(legend.position = "top")

### specify cross-validation to tune mtry
ctrl_k05 <- trainControl(method = "cv", number = 5)

### use cross-validation to tune mtry
set.seed(12341)
fit_rf_cv05 <- train(compressive_strength ~ .,
                     data = my_concrete,
                     method = "rf",
                     metric = "RMSE",
                     trControl = ctrl_k05,
                     tuneGrid = expand.grid(mtry = seq(2, 8, by = 1)),
                     importance = TRUE)

### print out the results
fit_rf_cv05

### look at the RMSE summaries including the standard error on the averages
fit_rf_cv05$results %>% tibble::as_tibble() %>% 
  ggplot(mapping = aes(x = mtry)) +
  geom_linerange(mapping = aes(ymin = RMSE - RMSESD/sqrt(5),
                               ymax = RMSE + RMSESD/sqrt(5),
                               group = mtry),
                 size = 1.2) +
  geom_point(mapping = aes(y = RMSE),
             size = 5) +
  theme_bw()

### now fit a boosted tree with GBM, use the default tuning
### grid for now
set.seed(12341)
fit_gbm_cv05 <- train(compressive_strength ~ .,
                      data = my_concrete,
                      method = "gbm",
                      metric = "RMSE",
                      trControl = ctrl_k05,
                      verbose=FALSE)

fit_gbm_cv05

plot(fit_gbm_cv05)

### maybe more trees (iterations) would improve performance?

fit_gbm_cv05$bestTune

### let's fix interaction depth, but try out more iterations and 
### a smaller learning rate
gbm_grid <- expand.grid(n.trees = c(100, 150, 300, 500, 750, 1000),
                        shrinkage = c(0.01, 0.1),
                        interaction.depth = fit_gbm_cv05$bestTune$interaction.depth,
                        n.minobsinnode = fit_gbm_cv05$bestTune$n.minobsinnode)

set.seed(12341)
fit_gbm_cv05_tune <- train(compressive_strength ~ .,
                           data = my_concrete,
                           method = "gbm",
                           metric = "RMSE",
                           tuneGrid = gbm_grid,
                           trControl = ctrl_k05)

fit_gbm_cv05_tune$bestTune

plot(fit_gbm_cv05_tune)

### now fit a boosted tree model with XGBOOST, will get a weird looking warning
### so include the additional objective argument to remove it, but you can
### ignore the warning even if you don't include this argument
set.seed(12341)
fit_xgb_cv05 <- train(compressive_strength ~ .,
                      data = my_concrete,
                      method = "xgbTree",
                      metric = "RMSE",
                      trControl = ctrl_k05,
                      objective = 'reg:squarederror')

### print out the results - there are lot of tuning parameter combinations!
fit_xgb_cv05

### visualize the RMSE cross-validation performance per tuning parameter
plot(fit_xgb_cv05)

fit_xgb_cv05$bestTune

### tune xgboost further focusing on the max depth and number of trees
xgb_grid <- expand.grid(nrounds = seq(100, 1100, by = 200),
                        max_depth = c(2, 4, 6, 8),
                        eta = c(0.1 * fit_xgb_cv05$bestTune$eta, 
                                0.5 * fit_xgb_cv05$bestTune$eta,
                                fit_xgb_cv05$bestTune$eta),
                        gamma = fit_xgb_cv05$bestTune$gamma,
                        colsample_bytree = fit_xgb_cv05$bestTune$colsample_bytree,
                        min_child_weight = fit_xgb_cv05$bestTune$min_child_weight,
                        subsample = fit_xgb_cv05$bestTune$subsample)

xgb_grid %>% dim()

set.seed(12341)
fit_xgb_cv05_tune <- train(compressive_strength ~ .,
                           data = my_concrete,
                           method = "xgbTree",
                           metric = "RMSE",
                           tuneGrid = xgb_grid,
                           trControl = ctrl_k05,
                           objective = 'reg:squarederror')

### the best tuning parameters
fit_xgb_cv05_tune$bestTune

### visualize the resampling performance
plot(fit_xgb_cv05_tune)

### compare the performance of the tuned random forest model with the
### tuned boosted trees, however let's have some context
### by comparing the performance relative to a simpler linear model
### fit a regularized model with elastic net penalty of a linear
### model with all pair-wise interactions

### notice that unlike the tree-based methods the inputs are preprocessed
### by calling the `preProcess` argument

### just use a default tuning grid for elastic net
set.seed(12341)
fit_glmnet_pairs <- train(compressive_strength ~ (.)^2, 
                          data = my_concrete,
                          method = "glmnet",
                          metric = "RMSE",
                          preProcess = c("center", "scale"),
                          trControl = ctrl_k05)

fit_glmnet_pairs

### let's now compare all models
concrete_model_compare <- resamples(list(GLMNET = fit_glmnet_pairs,
                                         RF = fit_rf_cv05,
                                         XGB = fit_xgb_cv05,
                                         XGBtune = fit_xgb_cv05_tune,
                                         GBM = fit_gbm_cv05,
                                         GBMtune = fit_gbm_cv05_tune))

dotplot(concrete_model_compare, metric = "RMSE")

### extract all of the resample fold performance metrics per model
concrete_model_compare_lf <- concrete_model_compare$values %>% tibble::as_tibble() %>% 
  pivot_longer(!c("Resample")) %>% 
  tidyr::separate(name,
                  c("model_name", "metric_name"),
                  sep = "~")

### visualize the performance metrics sumamries, show the individual
### fold results with markers and the cross-validation average
### and standard error, which model is better?
concrete_model_compare_lf %>% 
  ggplot(mapping = aes(x = model_name, y = value)) +
  geom_point() +
  stat_summary(fun.data = "mean_se",
               color = "red",
               fun.args = list(mult = 1)) +
  coord_flip() +
  facet_grid( . ~ metric_name, scales = "free_x") +
  theme_bw()

### show just the tree based methods
concrete_model_compare_lf %>% 
  filter(model_name != "GLMNET") %>% 
  ggplot(mapping = aes(x = model_name, y = value)) +
  geom_point() +
  stat_summary(fun.data = "mean_se",
               color = "red",
               fun.args = list(mult = 1)) +
  coord_flip() +
  facet_grid( . ~ metric_name, scales = "free_x") +
  theme_bw()

### variable importances
varImp(fit_rf_cv05)

varImp(fit_xgb_cv05_tune)

### -- ### -- ###

### binary classification example with IONOSPHERE data set
data("Ionosphere", package = "mlbench")

### remove the first 2 variables, turn the Class into a factor
my_ion <- Ionosphere %>% 
  select(-V1, -V2) %>% 
  mutate(Class = factor(Class, levels = c("good", "bad")))

my_ion %>% dim()

### set the grid of mtry values to consider
rf_grid <- expand.grid(mtry = seq(2, 30, by = 4))

### identify the optimal value of mtry based on accuracy
set.seed(4321)
fit_rf_ion <- train(Class ~ ., data = my_ion,
                    method = "rf",
                    metric = "Accuracy",
                    trControl = ctrl_k05,
                    tuneGrid = rf_grid,
                    importance = TRUE)

### print the results
fit_rf_ion

### how does the tuned value compare to the recommended value
### for a classification problem?
sqrt(32)

### plot the results
plot(fit_rf_ion)

### why is it better to use a small value for mtry even though
### there are over 30 inputs? consider the correlations between
### inputs!
my_ion %>% 
  select(-Class) %>% 
  cor() %>% 
  corrplot::corrplot(type = "upper", method = "square")

### reorder the inputs to make it easier to see highly correlated
### inputs grouped together
my_ion %>% 
  select(-Class) %>% 
  cor() %>% 
  corrplot::corrplot(type = "upper", method = "square",
                     order = "hclust", hclust.method = 'ward.D2')

### scatter plot between two of teh inputs as a check,
### include the best fit line between as a check
my_ion %>% 
  ggplot(mapping = aes(x = V21, y = V13)) +
  geom_point() +
  geom_smooth(method = "lm",
              formula = y ~ x) +
  theme_bw()

### accuracy performance metrics
confusionMatrix.train(fit_rf_ion)

### changing from Accuracy to ROC - maximizing the AUC
### need to create a new trainControl function call
### and include specifics for the ROC curve
ctrl_k05_roc <- trainControl(method = "cv", number = 5,
                             summaryFunction = twoClassSummary,
                             classProbs = TRUE,
                             savePredictions = TRUE)

### fit the random forest model
set.seed(4321)
fit_rf_ion_roc <- train(Class ~ ., data = my_ion,
                        method = "rf",
                        metric = "ROC",
                        trControl = ctrl_k05_roc,
                        tuneGrid = rf_grid,
                        importance = TRUE)

### print the results
fit_rf_ion_roc

### plot the results...is the behavior different
### from when we were focused on accuracy?
plot(fit_rf_ion_roc)

### check the accuracy
confusionMatrix.train(fit_rf_ion_roc)

### plot the ROC curve, first look at how the predictions
### on the fold test sets are structured
fit_rf_ion_roc$pred %>% tibble::as_tibble()

### use yardstick to simplify creating the roc curve
library(yardstick)

### ROC curve averaged over the folds for each mtry value
fit_rf_ion_roc$pred %>% tibble::as_tibble() %>% 
  group_by(mtry) %>% 
  roc_curve(obs, good) %>% 
  autoplot()

### focus on just a few of the mtry values
fit_rf_ion_roc$pred %>% tibble::as_tibble() %>% 
  filter(mtry %in% c(2, 10, 18, 26)) %>% 
  group_by(mtry) %>% 
  roc_curve(obs, good) %>% 
  autoplot()

### look at the ROC curve per fold
fit_rf_ion_roc$pred %>% tibble::as_tibble() %>% 
  filter(mtry %in% c(2, 10, 18, 26)) %>% 
  group_by(mtry, Resample) %>% 
  roc_curve(obs, good) %>% 
  ggplot(mapping = aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(mapping = aes(group = interaction(mtry, Resample),
                          color = Resample),
            size = 1.25, linetype = 'dashed') +
  geom_abline(slope = 1, intercept = 0, linetype = 'dotted') +
  coord_equal() +
  facet_wrap(~mtry, labeller = "label_both") +
  theme_bw() +
  theme(legend.position = "top")

### include the reasmple averaged ROC curve per mtry value
### to consider the averaged result and the variability in the perforamnce
fit_rf_ion_roc$pred %>% tibble::as_tibble() %>% 
  filter(mtry %in% c(2, 10, 18, 26)) %>% 
  group_by(mtry, Resample) %>% 
  roc_curve(obs, good) %>% 
  ggplot(mapping = aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(mapping = aes(group = interaction(mtry, Resample),
                          color = Resample),
            size = 1.25, linetype = 'dashed') +
  geom_path(data = fit_rf_ion_roc$pred %>% tibble::as_tibble() %>% 
              filter(mtry %in% c(2, 10, 18, 26)) %>% 
              group_by(mtry) %>% 
              roc_curve(obs, good),
            mapping = aes(group = mtry),
            color = 'black', size = 1.15) +
  geom_abline(slope = 1, intercept = 0, linetype = 'dotted') +
  coord_equal() +
  facet_wrap(~mtry, labeller = "label_both") +
  theme_bw() +
  theme(legend.position = "top")

### variable importance rankings, check when variables are the most
### important
plot(varImp(fit_rf_ion_roc))

### use a boosted tree classifer with GBM, continue to maximize the
### AUC and use the default tuning grid
set.seed(4321)
fit_gbm_ion_roc <- train(Class ~ .,
                         data = my_ion,
                         method = "gbm",
                         metric = "ROC",
                         trControl = ctrl_k05_roc)

fit_gbm_ion_roc

plot(fit_gbm_ion_roc)

### use a boosted tree classifier with xgboost, continue to maximize
### the AUC and use the default tuning grid for now
set.seed(4321)
fit_xgboost_ion_roc <- train(Class ~ ., 
                             data = my_ion,
                             method = "xgbTree",
                             metric = "ROC",
                             trControl = ctrl_k05_roc)

### print the results
fit_xgboost_ion_roc

### plot the results
plot(fit_xgboost_ion_roc)

### check the accuracy
confusionMatrix.train(fit_xgboost_ion_roc)

### before comparing random forst and boosted trees, fit a regularized
### logistic regression model with elastic net penalty for comparison
### it's always important to consider a simpler method to check if the
### extra complexity of the more advanced models is worth it!

### check cna we use pair-wise interactions????
model.matrix( Class ~ (.)^2, data = my_ion) %>% dim()

### use a custom grid this time for elastic net
enet_grid <- expand.grid(alpha = c(0.1, 0.2, 0.3, 0.4),
                         lambda = exp(seq(-6, 1, length.out = 21)))

set.seed(4321)
fit_glmnet_roc <- train(Class ~ ., data = my_ion,
                        method = "glmnet",
                        metric = "ROC",
                        tuneGrid = enet_grid,
                        preProc = c("center", "scale"),
                        trControl = ctrl_k05_roc)

plot(fit_glmnet_roc, xTrans = log)

fit_glmnet_roc$bestTune

### now compare all 3 models based on the ROC curve metrics
ion_model_compare <- resamples(list(GLMNET = fit_glmnet_roc,
                                    RF = fit_rf_ion_roc,
                                    XGB = fit_xgboost_ion_roc,
                                    GBM = fit_gbm_ion_roc))

### can use the default plot method to compare
dotplot(ion_model_compare)

### or extract the results and and create a custom figure
ion_model_compare_lf <- ion_model_compare$values %>% tibble::as_tibble() %>% 
  pivot_longer(!c("Resample")) %>% 
  tidyr::separate(name,
                  c("model_name", "metric_name"),
                  sep = "~")

### which model do you think is betteR?
ion_model_compare_lf %>% 
  ggplot(mapping = aes(x = model_name, y = value)) +
  geom_point() +
  stat_summary(fun.min = 'min', fun.max='max',
               geom = 'linerange',
               color = 'grey') +
  stat_summary(fun.data = "mean_se",
               color = "red",
               fun.args = list(mult = 1)) +
  facet_grid( . ~ metric_name) +
  theme_bw()

### show just the tree based methods
ion_model_compare_lf %>% 
  filter(model_name != "GLMNET") %>% 
  ggplot(mapping = aes(x = model_name, y = value)) +
  geom_point() +
  stat_summary(fun.min = 'min', fun.max='max',
               geom = 'linerange',
               color = 'grey') +
  stat_summary(fun.data = "mean_se",
               color = "red",
               fun.args = list(mult = 1)) +
  facet_grid( . ~ metric_name) +
  theme_bw()
