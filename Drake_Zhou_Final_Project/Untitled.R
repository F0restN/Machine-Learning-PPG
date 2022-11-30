fit_and_assess <- function(mod, test_data, y_name)
{
  pred_test <- as.vector(predict(mod, newdata = test_data))
  
  y_test <- test_data %>% dplyr::select(all_of(y_name)) %>% pull()
  
  test_metrics <- tibble::tibble(
    rmse_value = rmse_vec(y_test, pred_test),
    mae_value = mae_vec(y_test, pred_test),
    r2_value = rsq_vec(y_test, pred_test)
  )
  
  return(test_metrics)
}



my_pred_all_cls %>%
  group_by(xn_05, customer) %>%
  arrange(desc(pred_prob), group_by = customer) %>%
  summarise(num_post = n(),
            trend_avg = mean(pred_prob),
            trend_lwr = quantile(pred_prob, 0.05),
            trend_upr = quantile(pred_prob, 0.95)) %>%
  ggplot(mapping = aes(x = xn_06)) +
  # geom_ribbon(mapping = aes(ymin = y_lwr, ymax = y_upr,
  #                           group = xb_07), 
  #             fill = "darkorange") +
  geom_ribbon(mapping = aes(ymin = trend_lwr, ymax = trend_upr,
                            group = xb_07),
              fill = "grey") +
  geom_line(mapping = aes(y = trend_avg,
                          group = xb_07),
            color = "black", size = 0.85) +
  facet_wrap(~xb_07, labeller = "label_both", scales = 'free') +
  labs(y = "y") +
  theme_bw()
# pivot_longer(!c(customer))
# %>% left_join(viz_input_grid_cls 