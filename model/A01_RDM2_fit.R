rm(list=ls()) 
library(dplyr)
library(tidyr)
library(rstan)

options(mc.cores=parallel::detectCores(4))

# first column = chosen rating 
# second-last column = other ratings
makeRatingMatrix = function(rating_p1) {
  if (rating_p1 %% 2 == 1) { # odd numbers
    ratingVec = c(1,3,5,7,9)
  } else if (rating_p1 %% 2 == 0) {
    ratingVec = c(2,4,6,8,10)}
  return(c(rating_p1, ratingVec[ratingVec != rating_p1]))
}


load("../data/behav_eye_combined_excludeFoodOnly.RData")
rm(list = c("behavEye2", "eye_exc3"))

# exclude problematic subject 
behav_exc3 = behav_exc3 %>% filter(id != 12)

data_wide = behav_exc3 %>% 
  filter(phase == 1) %>% 
  arrange(id, trial) %>%
  select(id, trial, foodIndex, rating, RT1) %>%
  rename(rating_p1 = rating) %>% 
  right_join(behav_exc3 %>% filter(phase == 2) %>%
               select(id, foodIndex, rating) %>% 
               rename(rating_p2 = rating), 
             by = c("id", "foodIndex")) %>% 
  mutate(trialType = ifelse((rating_p1 %% 2) == (rating_p2 %% 2), "available", "unavailable"), 
         trialType = factor(trialType, levels = c("available", "unavailable")))

id_idx = data.frame(id = unique(data_wide$id)) %>%
  mutate(subj_idx = 1:n())

data_wide = data_wide %>% left_join(id_idx, by = "id")

# 
S = nrow(id_idx)
subj = data_wide$subj_idx

# i
rating_raw = t(sapply(data_wide$rating_p1, makeRatingMatrix))

# rating was re-scaled to 0 and 1 for efficient sampling in stan 
# scaled |i - rating2|
rating_scaled = abs(rating_raw/10 - data_wide$rating_p2/10)

numBeta = 3
# prepare data for stan 
dataList = list(
  N = nrow(rating_scaled), # number of trial
  S = S, # number of subjects
  subj = subj, 
  Kv = numBeta, # number of betas in drift
  Kx = 5, # number of accumulator
  x_chosen = rating_scaled[,1], 
  x_unchosen = rating_scaled[,2:5],
  x_strength = abs(data_wide$rating_p2 - 5.5)/10, # preference strength, abs(r2 - 5.5)
  subj_start = sapply(1:S, function(x) min(which(subj == x))),
  subj_end = sapply(1:S, function(x) max(which(subj == x))),
  RT = data_wide$RT1
)

# stan_model = stan_model("RDM2.stan")
# saveRDS(stan_model, "RDM2_compiled.RData")
stan_model = readRDS("RDM2_compiled.RData")

initf <- function() {
  list(a_mu_raw = log(2), 
       a_raw = rep(0, each=S),
       a = rep(2, each=S),
       v_beta_mu = rep(0, numBeta), 
       v_beta_raw = matrix(0, numBeta, S), 
       v_beta = matrix(0, numBeta, S))
}

model_obj = rstan::sampling(stan_model,
                            data = dataList, 
                            pars = c("a_mu_raw", "a_sd", "a",
                                     "v_beta_mu", "v_beta_sd", "v_beta",
                                     "log_lik", "dev"), 
                            init = initf,
                            warmup = 1000, iter = 3000, chains = 4, cores = 4, 
                            control=list(adapt_delta=0.99,
                                         stepsize = 0.01, max_treedepth = 15))

save(dataList, data_wide, file = "RDM2_hier_data.RData")
saveRDS(model_obj, "RDM2_hier_fit.RData")
