# The influence of attention on the preference rating process


- [model](model) contains codes to fit models and generate posterior predictions
    - `*_fit.R`: R code to fit a model 
    - `*.stan`: Stan code for a model 
    - `*_ppd.R`: R code to generate predictions using posterior samples

- [model_evaluation](model_evaluation) contains R markdown files for comparing model performance and diagnosing posterior draws
    - `A02_model_comparison`: Compare model performance using LOOIC 
    - `A02_RDM6_posterior`: Check R hat values, effective sample size, trace plots, and posterior distributions