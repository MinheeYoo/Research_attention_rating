rm(list=ls()) 

library(rstan)
library(dplyr)
library(tidyr)

model_obj = readRDS('RDM5_hier_fit.RData')
load('RDM5_hier_data.RData')

# sample index
nSim = 1000
set.seed(0)
sample_idx = sort(sample(1:8000, nSim))

a_samples = as.data.frame(model_obj, pars = "a") %>%
  mutate(draw = 1:n()) %>%
  filter(draw %in% sample_idx) %>% 
  relocate(draw) %>% arrange(draw)

theta_samples = as.data.frame(model_obj, pars = "theta") %>%
  mutate(draw = 1:n()) %>%
  filter(draw %in% sample_idx) %>% 
  relocate(draw) %>% arrange(draw)

tmp = as.data.frame(model_obj, pars = "v_beta") %>%
  mutate(draw = 1:n()) %>%
  filter(draw %in% sample_idx) %>% 
  pivot_longer(!draw, 
               names_to = "tmpName", 
               values_to = "sampleValue") %>% 
  mutate(param_idx = as.numeric(substr(tmpName, 8, 8)),
         subj_idx =  as.numeric(substr(tmpName, 10, nchar(tmpName)-1))) 

v1_samples = tmp %>% 
  filter(param_idx == 1) %>% 
  select(!c(tmpName, param_idx)) %>% 
  arrange(subj_idx) %>%
  pivot_wider(names_from = subj_idx, 
              values_from = sampleValue) %>% 
  arrange(draw)

v2_samples = tmp %>% 
  filter(param_idx == 2) %>% 
  select(!c(tmpName, param_idx)) %>% 
  arrange(subj_idx) %>%
  pivot_wider(names_from = subj_idx, 
              values_from = sampleValue) %>%
  arrange(draw)

rm(tmp)

# generate predictions 
# first column = chosen rating 
# second-last column = other ratings
makeRatingMatrix = function(rating_p1) {
  if (rating_p1 %% 2 == 1) { # odd numbers
    ratingVec = c(1,3,5,7,9)
  } else if (rating_p1 %% 2 == 0) {
    ratingVec = c(2,4,6,8,10)}
  return(c(rating_p1, ratingVec[ratingVec != rating_p1]))
}

rating_raw = t(sapply(data_wide$rating_p1, makeRatingMatrix))

# simulate data with a posterior sample
sim_data = function(dataList, v1, v2, a, theta, rating_raw) {
  # v1, v2, a = vector[N]
  
  N = dataList$N
  x_v = cbind(dataList$x_chosen, dataList$x_unchosen)
  dwell = cbind(dataList$d_chosen, dataList$d_unchosen)
  
  ### initialize matrix
  v_trial = matrix(0, N, 5)
  a_trial = rep(0, N)
  fpt = matrix(0, N, 5)
  
  ### get trial level drift rate and boundary 
  a_trial = a[dataList$subj]
  for (i in 1:5){
    v_trial[,i] = exp(v1[dataList$subj] + 
                        v2[dataList$subj] * x_v[,i] +
                        theta[dataList$subj] * dwell[,i])
  }
  
  ### generate prediction with inv gaussian distribution
  # mu = a/v
  # lambda = a^2
  for (i in 1:5){
    fpt[, i] = statmod::rinvgauss(N, mean = a_trial/v_trial[,i], 
                                  shape = a_trial^2)
  }
  
  # rating and rt
  # first accumulator reaching boundary 
  min_idx = apply(fpt, 1, which.min)
  pred = data.frame(
    rating_pred = sapply(1:N, function(x) rating_raw[x, min_idx[x]]),
    rt_pred = apply(fpt, 1, min)) 
  
  return(pred)
}


# generate predictions 
ppd = data.frame()
for (i in 1:nSim) {
  tmp = sim_data(dataList, 
                 v1 = unlist(v1_samples[i, 2:47]), 
                 v2 = unlist(v2_samples[i, 2:47]), 
                 a = unlist(a_samples[i, 2:47]), 
                 theta = unlist(theta_samples[i, 2:47]),
                 rating_raw)
  ppd = rbind(ppd, 
              data.frame(sim_idx = i,
                         trial_idx = 1:nrow(tmp), 
                         tmp))
}

save(ppd, file = "RDM5_hier_ppd.RData")



