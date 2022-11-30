### code associated with categorical inputs slides

library(tidyverse)

library(modelr)

sim2

sim2 %>% 
  ggplot(mapping = aes(x = x, y = y)) +
  geom_point(size = 3) +
  theme_bw()

### fit the linear model
mod_02 <- lm(y ~ x, data = sim2)

mod_02 %>% summary()

### show the design matrix
Xmat_02 <- model.matrix(y ~ x, data = sim2)

head(Xmat_02)

head(sim2)

### look at when x == "b"
Xmat_02[8:12, ]

sim2 %>% slice(8:12)

### make predictions on the training set
pred_train_02 <- predict(mod_02, sim2)

sim2 %>% 
  mutate(ypred = pred_train_02) %>% 
  ggplot(mapping = aes(x = x)) +
  geom_point(mapping = aes(y = y),
             size = 3) +
  geom_point(mapping = aes(y = ypred),
             color = "red", size = 4) +
  theme_bw()

### double check by calculating the average response
sim2 %>% 
  mutate(ypred = pred_train_02) %>% 
  ggplot(mapping = aes(x = x)) +
  geom_point(mapping = aes(y = y),
             size = 3) +
  stat_summary(fun.y = "mean",
               geom = "point",
               mapping = aes(y = y),
               size = 7, shape = 15, 
               color = "steelblue") +
  geom_point(mapping = aes(y = ypred),
             color = "red", size = 4) +
  theme_bw()

### look at the coefficient MLEs
coef(mod_02)

### compare to the grouped averages
sim2 %>% 
  group_by(x) %>% 
  summarise(num_rows = n(),
            y_avg = mean(y)) %>% 
  ungroup()

### coefficient summaries for the dummary variable coefficients
coefplot::coefplot(mod_02) + 
  coord_cartesian(xlim = c(-0.15, 8.55)) +
  labs(title = "DUMMY VARIABLE coefficient plot") +
  theme_bw() +
  theme(legend.position = 'none')

### build a model without the intercept
mod_no_int <- lm(y ~ x - 1, data = sim2)

mod_no_int %>% summary()

model.matrix(y ~ x - 1, data = sim2) %>% head()

### compare coefficient estimates associated with
### the one-hot encoding with the group average values
sim2 %>% 
  group_by(x) %>% 
  summarise(y_avg = mean(y)) %>% 
  mutate(one_hot_coef = as.vector( coef(mod_no_int) ))

### coefficient summaries for the one-hot encoding model
coefplot::coefplot(mod_no_int) + 
  coord_cartesian(xlim = c(-0.15, 8.55)) +
  labs(title = "ONE-HOT ENCODING coefficient plot") +
  theme_bw() +
  theme(legend.position = 'none')
