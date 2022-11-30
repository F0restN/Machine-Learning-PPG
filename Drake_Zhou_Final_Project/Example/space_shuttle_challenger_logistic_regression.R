### demonstrate bayesian logistic regression using real data
### the space shuttle challenger data set from the 
### Bayesian methods for Hackers book:
### https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers

library(tidyverse)

### read in the data

shuttle_url <- 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter2_MorePyMC/data/challenger_data.csv'

challenger_data <- readr::read_csv(shuttle_url, col_names = TRUE)

challenger_data

challenger_data %>% tail() ### last observation is the accident

### remove the missing and last observation, change the data type
### of the `Damage Incident` variable and rename

clean_df <- challenger_data %>% 
  filter(`Damage Incident` %in% c('0', '1')) %>% 
  select(Date, Temperature, outcome = `Damage Incident`) %>% 
  mutate_at(c("outcome"), as.numeric)

clean_df

### plot the training data
clean_df %>% 
  ggplot(mapping = aes(x = Temperature, y = outcome)) +
  geom_jitter(height = 0.05, size = 4.5, alpha = 0.5) +
  theme_bw()

### standardize the temperature
train_df <- clean_df %>% 
  mutate(x = (Temperature - mean(Temperature))/sd(Temperature))

### define the log-posterior function

logistic_logpost <- function(unknowns, my_info)
{
  # unpack the parameter vector
  beta_0 <- unknowns[1]
  beta_1 <- unknowns[2]
  
  # calculate linear predictor
  eta <- beta_0 + beta_1 * my_info$xobs
  
  # calculate the event probability
  mu <- boot::inv.logit(eta)
  
  # evaluate the log-likelihood
  log_lik <- sum(dbinom(x = my_info$yobs,
                        size = 1,
                        prob = mu,
                        log = TRUE))
  
  # evaluate the log-prior
  log_prior <- sum(dnorm(x = c(beta_0, beta_1),
                         mean = my_info$mu_beta,
                         sd = my_info$tau_beta,
                         log = TRUE))
  
  # sum together
  log_lik + log_prior
}

### define the list of required information using the a regularizing
### prior...prior standard deviation of 3
info_use <- list(
  xobs = train_df$x,
  yobs = train_df$outcome,
  mu_beta = 0,
  tau_beta = 3
)

### define the grid of intercept and slope values
beta_grid <- expand.grid(beta_0 = seq(-4, 4, length.out = 251),
                         beta_1 = seq(-4, 4, length.out = 251),
                         KEEP.OUT.ATTRS = FALSE,
                         stringsAsFactors = FALSE) %>% 
  as.data.frame() %>% tibble::as_tibble()

### calculate the log-posterior over the grid of parameter values
### and visualize the log-posterior surface
beta_grid %>% 
  rowwise() %>% 
  mutate(log_post = logistic_logpost(c(beta_0, beta_1), info_use)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max(log_post)) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  geom_raster(mapping = aes(fill = log_post_2)) +
  stat_contour(mapping = aes(z = log_post_2),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 1.05,
               color = "black") +
  coord_fixed(ratio = 1) +
  scale_fill_viridis_c(guide = 'none', option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))

### compare to a weakly infomrative prior with prior sd of 25
info_weak <- list(
  xobs = train_df$x,
  yobs = train_df$outcome,
  mu_beta = 0,
  tau_beta = 25
)

### calculate the log-posterior based on the weak prior over
### the grid of parameter values and visualize the posterior surface
beta_grid %>% 
  rowwise() %>% 
  mutate(log_post = logistic_logpost(c(beta_0, beta_1), info_weak)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max(log_post)) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  geom_raster(mapping = aes(fill = log_post_2)) +
  stat_contour(mapping = aes(z = log_post_2),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 1.05,
               color = "black") +
  coord_fixed(ratio = 1) +
  scale_fill_viridis_c(guide = 'none', option = "viridis",
                       limits = log(c(0.01/100, 1.0))) +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))

### compare the regularizing and the weak priors
beta_grid %>% 
  rowwise() %>% 
  mutate(log_post = logistic_logpost(c(beta_0, beta_1), info_use)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max(log_post)) %>% 
  mutate(prior_sd = 3) %>% 
  bind_rows(beta_grid %>% 
              rowwise() %>% 
              mutate(log_post = logistic_logpost(c(beta_0, beta_1), info_weak)) %>% 
              ungroup() %>% 
              mutate(log_post_2 = log_post - max(log_post)) %>% 
              mutate(prior_sd = 25)) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  geom_hline(yintercept = 0, color = 'grey', linetype = 'dashed', size=1.1) +
  geom_vline(xintercept = 0, color = 'grey', linetype = 'dashed', size=1.1) +
  stat_contour(mapping = aes(z = log_post_2,
                             color = as.factor(prior_sd)),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 1.25) +
  coord_fixed(ratio = 1) +
  ggthemes::scale_color_excel_new(theme = 'Office Theme', "prior standard deviation") +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))


### perform the laplace approximation

my_laplace <- function(start_guess, logpost_func, ...)
{
  # code adapted from the `LearnBayes`` function `laplace()`
  fit <- optim(start_guess,
               logpost_func,
               gr = NULL,
               ...,
               method = "BFGS",
               hessian = TRUE,
               control = list(fnscale = -1, maxit = 1001))
  
  mode <- fit$par
  post_var_matrix <- -solve(fit$hessian)
  p <- length(mode)
  int <- p/2 * log(2 * pi) + 0.5 * log(det(post_var_matrix)) + logpost_func(mode, ...)
  # package all of the results into a list
  list(mode = mode,
       var_matrix = post_var_matrix,
       log_evidence = int,
       converge = ifelse(fit$convergence == 0,
                         "YES", 
                         "NO"),
       iter_counts = as.numeric(fit$counts[1]))
}

### execute the laplace approximation using the regularizing prior
### does the starting guess matter?????
shuttle_laplace <- my_laplace(rep(0, 2), logistic_logpost, info_use)

shuttle_laplace

### posterior means on the intercept and slope
shuttle_laplace$mode

### posterior standard deviations on the parameters
shuttle_laplace$var_matrix %>% diag() %>% sqrt()

### 95% uncertainty interval on the slope
c(shuttle_laplace$mode[1+1] - 2 * sqrt(diag(shuttle_laplace$var_matrix)[1+1]),
  shuttle_laplace$mode[1+1] + 2 * sqrt(diag(shuttle_laplace$var_matrix)[1+1]))

### probability that the slope is greater than 0?
1 - pnorm(0, mean = shuttle_laplace$mode[1+1], sd = sqrt(diag(shuttle_laplace$var_matrix)[1+1]))

### generate posteiror samples

generate_glm_post_samples <- function(mvn_result, num_samples)
{
  # specify the number of unknown beta parameters
  length_beta <- length(mvn_result$mode)
  
  # generate the random samples
  beta_samples <- MASS::mvrnorm(n = num_samples, 
                                mu = mvn_result$mode, 
                                Sigma = mvn_result$var_matrix)
  
  # change the data type and name
  beta_samples %>% 
    as.data.frame() %>% tibble::as_tibble() %>% 
    purrr::set_names(sprintf("beta_%02d", (1:length_beta) - 1))
}

### generate the posterior samples
set.seed(912312)
post_betas <- generate_glm_post_samples(shuttle_laplace, 1e4)

### visualize the posterior density
post_betas %>% 
  ggplot(mapping = aes(x = beta_01, y = beta_00)) +
  geom_density2d_filled() +
  geom_density2d(color = "white") +
  theme_bw()

### visualize the posterior histograms
post_betas %>% tibble::rowid_to_column("post_id") %>% 
  pivot_longer(!c("post_id")) %>% 
  ggplot(mapping = aes(x = value)) +
  geom_histogram(bins = 55) +
  facet_wrap(~name, scales = "free_y") +
  theme_bw() +
  theme(axis.text.y = element_blank())

### posterior mean on the slope
mean( post_betas$beta_01 )

### posterior 95% uncertainty interval
quantile(post_betas$beta_01, c(0.025, 0.975))

### posterior probability that the slope is greater than 0 using the samples

mean( post_betas$beta_01 > 0 )

### compare the laplace appromation associated posterior samples with the
### true posterior
beta_grid %>% 
  rowwise() %>% 
  mutate(log_post = logistic_logpost(c(beta_0, beta_1), info_use)) %>% 
  ungroup() %>% 
  mutate(log_post_2 = log_post - max(log_post)) %>% 
  mutate(prior_sd = 3) %>% 
  ggplot(mapping = aes(x = beta_1, y = beta_0)) +
  geom_point(data = post_betas,
             mapping = aes(x = beta_01, y = beta_00),
             alpha = 0.2) +
  geom_hline(yintercept = 0, color = 'grey', linetype = 'dashed', size=1.1) +
  geom_vline(xintercept = 0, color = 'grey', linetype = 'dashed', size=1.1) +
  stat_contour(mapping = aes(z = log_post_2,
                             color = as.factor(prior_sd)),
               breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),
               size = 2.25) +
  coord_fixed(ratio = 1) +
  ggthemes::scale_color_excel_new(theme = 'Office Theme', "prior standard deviation") +
  labs(x = expression(beta[1]), y = expression(beta[0])) +
  theme_bw() +
  theme(legend.position = "top",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12))


### as a comparison, find the MLE to the coefficients

fit_glm <- glm(outcome ~ x, family = "binomial", data = train_df)

fit_glm %>% summary()

### posterior predictions on the event probability

### we know how to do this already! it's just matrix math!!

### But, let's iterate over the posterior samples so we can focus on
### the individual steps. For a given posterior sample of the coefficients
### we must calculate the linear predictor then backtransform to
### calculate the event probability

make_post_preds <- function(b0, b1, post_id, x)
{
  eta <- b0 + b1 * x
  
  mu <- boot::inv.logit(eta)
  
  list(eta = eta, mu = mu, 
       pred_id = seq_along(x),
       post_id = rep(post_id, length(mu)))
}

### define a prediction grid for the STANDARDIZED input

xviz <- seq(-3.5, 3.5, length.out = 251)

### use purrr to iterate over the posterior samples

post_pred_for_viz <- purrr::pmap_dfr(list(post_betas$beta_00,
                                          post_betas$beta_01,
                                          seq_along(post_betas$beta_00)),
                                     make_post_preds,
                                     x = xviz)

### lots of rows in the data set!!!!!
post_pred_for_viz %>% glimpse()

### calculate the posterior prediction summaries on the event probability
### for each prediction point (input location)

post_pred_summary <- post_pred_for_viz %>% 
  group_by(pred_id) %>% 
  summarise(num_post = n(),
            mu_avg = mean(mu),
            mu_med = median(mu),
            mu_q025 = as.vector( quantile(mu, 0.025) ),
            mu_q25 = as.vector( quantile(mu, 0.25) ),
            mu_q75 = as.vector( quantile(mu, 0.75) ),
            mu_q975 = as.vector( quantile(mu, 0.975) ),
            .groups = 'drop') %>% 
  left_join(tibble::tibble(x = xviz) %>% tibble::rowid_to_column("pred_id"),
            by = "pred_id")

### visualize the posterior prediction summaries
post_pred_summary %>% 
  ggplot(mapping = aes(x = x)) +
  # posterior middle 95% uncertainty interval
  geom_ribbon(mapping = aes(ymin = mu_q025,
                            ymax = mu_q975),
              fill = 'grey50', alpha = 0.3) +
  # posterior middle 50% uncertainty interval
  geom_ribbon(mapping = aes(ymin = mu_q25,
                            ymax = mu_q75),
              fill = 'grey50', alpha = 0.4) +
  # posterior median
  geom_line(mapping = aes(y = mu_med),
            color = "white", size = 1.1) +
  # posterior mean
  geom_line(mapping = aes(y = mu_avg),
            color = "black", size = 1.1) +
  # training data
  geom_point(data = train_df,
             mapping = aes(x = x, y = outcome),
             size = 7.5, alpha = 0.5, color = 'red') +
  labs(x = 'Standardized Temperature', y = 'event probability') +
  theme_bw()

### show the x-axis in terms of Temperature not the standardized input
### and denote the temperature on the day of the accident
post_pred_summary %>% 
  mutate(Temperature = sd(train_df$Temperature) * x + mean(train_df$Temperature)) %>% 
  ggplot(mapping = aes(x = Temperature)) +
  geom_ribbon(mapping = aes(ymin = mu_q025,
                            ymax = mu_q975),
              fill = 'grey50', alpha = 0.3) +
  geom_ribbon(mapping = aes(ymin = mu_q25,
                            ymax = mu_q75),
              fill = 'grey50', alpha = 0.4) +
  geom_line(mapping = aes(y = mu_med),
            color = "white", size = 1.1) +
  geom_line(mapping = aes(y = mu_avg),
            color = "black", size = 1.1) +
  geom_point(data = train_df,
             mapping = aes(x = Temperature, y = outcome),
             size = 7.5, alpha = 0.5, color = 'red') +
  geom_vline(data = challenger_data %>% 
               filter(`Damage Incident` == "Challenger Accident"),
             mapping = aes(xintercept = Temperature),
             color = 'black', size = 1.25) +
  labs(y = 'event probability') +
  theme_bw()

### as a comparison also include the predictions from the non-bayesian
### model, with the confidence interval on the event probability
### need to perform the backtransformation correctly!!!!

pred_mle_link <- predict(fit_glm, newdata = data.frame(x = xviz),
                         type = 'link', se.fit = TRUE)

viz_pred_mle <- tibble::tibble(
  x = xviz,
  fit = predict(fit_glm, newdata = data.frame(x = xviz), type = 'response'),
  lwr = fit_glm$family$linkinv(pred_mle_link$fit - 1.96 * pred_mle_link$se.fit),
  upr = fit_glm$family$linkinv(pred_mle_link$fit + 1.96 * pred_mle_link$se.fit)
)

post_pred_summary %>% 
  mutate(Temperature = sd(train_df$Temperature) * x + mean(train_df$Temperature)) %>% 
  ggplot(mapping = aes(x = Temperature)) +
  geom_ribbon(mapping = aes(ymin = mu_q025,
                            ymax = mu_q975),
              fill = 'grey50', alpha = 0.3) +
  geom_ribbon(mapping = aes(ymin = mu_q25,
                            ymax = mu_q75),
              fill = 'grey50', alpha = 0.4) +
  geom_line(mapping = aes(y = mu_med),
            color = "white", size = 1.1) +
  geom_line(mapping = aes(y = mu_avg),
            color = "black", size = 1.1) +
  geom_line(data = viz_pred_mle %>% 
              pivot_longer(!c("x")) %>% 
              mutate(line_type_use = ifelse(name == "fit",
                                            "fit",
                                            "conf_bound")) %>% 
              mutate(Temperature = sd(train_df$Temperature) * x + mean(train_df$Temperature)),
            mapping = aes(x = Temperature, 
                          y = value,
                          group = name,
                          linetype = line_type_use),
            size = 1.55, color = 'navyblue') +
  geom_point(data = train_df,
             mapping = aes(x = Temperature, y = outcome),
             size = 7.5, alpha = 0.5, color = 'red') +
  scale_linetype_manual("",
                        labels = c("fit" = "MLE predicted event probability",
                                   "conf_bound" = "95% confidence interval bound"),
                        values = c("fit" = 'dashed',
                                   'conf_bound' = 'dotted')) +
  labs(x = 'Temperature', y = 'event probability') +
  theme_bw() +
  theme(legend.position = "top")

### probabilistic question: what is the probability that the event probability
### is greater than 50%???? what about 25%? Or 12.5%?
post_pred_summary_b <- post_pred_for_viz %>% 
  group_by(pred_id) %>% 
  summarise(num_post = n(),
            mu_avg = mean(mu),
            mu_med = median(mu),
            mu_q025 = as.vector( quantile(mu, 0.025) ),
            mu_q975 = as.vector( quantile(mu, 0.975) ),
            prob_grt_50 = mean(mu > 0.5),
            prob_grt_25 = mean(mu > 0.25),
            prob_grt_12.5 = mean(mu > 0.125),
            .groups = 'drop') %>% 
  left_join(tibble::tibble(x = xviz) %>% tibble::rowid_to_column("pred_id"),
            by = "pred_id")

post_pred_summary_b %>% 
  mutate(Temperature = sd(train_df$Temperature) * x + mean(train_df$Temperature)) %>% 
  select(pred_id, Temperature, starts_with("prob_grt_")) %>% 
  pivot_longer(starts_with("prob_grt_")) %>% 
  tidyr::separate(name,
                  c("prob_word", "grt_word", "threshold"),
                  sep = "_") %>% 
  mutate(threshold = as.numeric(threshold)) %>% 
  ggplot(mapping = aes(x = Temperature)) +
  geom_line(mapping = aes(y = value,
                          color = as.factor(threshold)),
            size = 1.35) +
  scale_color_viridis_d("threshold percent") +
  labs(y = 'posterior probability the event probability is greater than a threshold') +
  theme_bw() +
  theme(legend.position = 'top')
