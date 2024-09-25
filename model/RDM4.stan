functions {
  real inv_gaussian_lpdf(real x, real v, real a){
    real temp1;
    real temp2;
    temp1 = a / (sqrt(2 * pi() * (x^3)));
    temp2 = (a - v * x)^2/(2 * x);
    return (log(temp1) - temp2);
  }
  
  real inv_gaussian_lccdf(real x, real v, real a){
    real temp1;
    real temp2;
    real temp3;
    temp1 = a / sqrt(x);
    temp2 = x * v / a;
    temp3 = 1 - Phi_approx(temp1 * (temp2 - 1)) - exp(2 * a * v) * Phi_approx(- temp1 * (temp2 + 1));
    return(log(temp3));
  }
  
  real inv_gaussian_rng(real v, real a){
    real mu;
    real lam;
    real y;
    real z;
    real sample;
    real x;
    mu = a/v;
    lam = a^2;
    y = normal_rng(0, 1);
    y = y^2;
    x = mu + (mu^2*y)/(2*lam) - mu/(2*lam)*sqrt(4*mu*lam*y + mu^2*y^2);
    z = uniform_rng(0, 1);
    if (z > mu/(mu+x)){
      sample = mu^2/x;
    }else{
      sample = x;
    }
    return(sample);
  }
  
  real RDM_lpdf(real RT, real a, real v_chosen, row_vector v_unchosen){
    vector[5] logLik;
    
    // f(x) of chosen rating
    logLik[1] = inv_gaussian_lpdf(RT | v_chosen, a);
    // 1-F(x) of unchosen rating
    for (i in 1:4) logLik[i+1] = inv_gaussian_lccdf(RT | v_unchosen[i], a);
    
    return(sum(logLik));
  }
  
  row_vector RDM_rng(real a, real v_chosen, row_vector v_unchosen) {
    vector[5] fpt;
    row_vector[2] pred;
    
    // f(x) for chosen rating
    fpt[1] = inv_gaussian_rng(v_chosen, a);
    // f(x) for unchosen rating
    for (i in 1:4) fpt[i+1] = inv_gaussian_rng(v_unchosen[i], a);
    
    // decide choice and RT 
    pred[1] = sort_indices_asc(fpt)[1]; // choice
    pred[2] = sort_asc(fpt)[1]; // RT
    return(pred);
  }
}


data {
  int<lower=0> N; // number of data points
  int<lower=0> S; // number of subjects 
  int<lower=0, upper=S> subj[N]; // subject index of each trial
  int Kv; // number of betas in drift 
  int Kx; // number of accumulators
  vector[N] x_chosen; // |i-r2| of chosen rating 
  matrix[N, (Kx-1)] x_unchosen; // |i-r2| of unchosen rating 
  vector[N] x_strength; // strength of preference |r2-5.5|
  vector[N] d_chosen; // dwell time of chosen rating 
  matrix[N, (Kx-1)] d_unchosen; // dwell time of unchosen rating
  int subj_start[S]; 
  int subj_end[S]; 
  real RT[N];
}

parameters {
  // group level parameters 
  // boundary
  real a_mu_raw; 
  real<lower=0> a_sd;
  // drift 
  vector[Kv] v_beta_mu;
  vector<lower=0>[Kv] v_beta_sd;
  // theta = attentional discount 
  real theta_mu_raw; 
  real<lower=0> theta_sd; 
  // subject level parameters 
  vector[S] a_raw; 
  row_vector[S] v_beta_raw[Kv]; // matrix[Kv, S]
  vector[S] theta_raw; 
}

transformed parameters {
  vector[S] a; 
  matrix[Kv, S] v_beta; 
  vector<lower=0,upper=1>[S] theta; 
  
  vector[N] v_chosen_tmp; 
  matrix[N, (Kx-1)] v_unchosen_tmp;
  
  vector[N] v_chosen;
  matrix[N, (Kx-1)] v_unchosen; 
  
  // define individual level parameter
  // boundary
  a = exp(a_mu_raw + a_sd * a_raw);
  
  // coefficients on the drift 
  for (i in 1:Kv) v_beta[i, ] = v_beta_mu[i] + v_beta_sd[i] * v_beta_raw[i]; // v_beta_raw[i] = ith row
  
  // theta = attentional discount 
  theta = inv_logit(theta_mu_raw + theta_sd * theta_raw); 
  
  // tmp drift rate variable 
  // E_i = sum of v_i * x_i
  for (s in 1:S){
    // chosen rating
    v_chosen_tmp[subj_start[s]:subj_end[s]] = 
      v_beta[1,s] 
      + v_beta[2,s] * x_chosen[subj_start[s]:subj_end[s]]
      + v_beta[3,s] * x_strength[subj_start[s]:subj_end[s]];
    // unchosen rating
    for (i in 1:(Kx-1)){
      v_unchosen_tmp[subj_start[s]:subj_end[s],i] = 
        v_beta[1,s] 
        + v_beta[2,s] * x_unchosen[subj_start[s]:subj_end[s],i]
        + v_beta[3,s] * x_strength[subj_start[s]:subj_end[s]];
    }
  }
  
  // drift rate 
  // exp(dwell_i*E_i + theta * (1-dwell_i) * E_i)
  for (n in 1:N) {
    // chosen rating
    v_chosen[n] = 
      exp(d_chosen[n] * v_chosen_tmp[n] + 
        theta[subj[n]] * (1 - d_chosen[n]) * v_chosen_tmp[n]);
    
    // unchosen rating
    for (i in 1:(Kx-1)){
      v_unchosen[n,i] = 
        exp(d_unchosen[n,i] * v_unchosen_tmp[n,i] +
          theta[subj[n]] * (1 - d_unchosen[n,i]) * v_unchosen_tmp[n,i]);
    }
  }
}

model {
  // Priors 
  // Group level parameters 
  // Mean
  a_mu_raw ~ std_normal(); 
  v_beta_mu ~ std_normal();
  theta_mu_raw ~ std_normal(); 
  
  // standard deviation
  a_sd ~ normal(0, 1)T[0,];
  for (i in 1:Kv) v_beta_sd[i] ~ normal(0, 1)T[0,]; 
  theta_sd ~ normal(0, 1)T[0,];
  
  // Subject level parameters
  a_raw ~ std_normal(); 
  for (i in 1:Kv) v_beta_raw[i] ~ std_normal(); 
  theta_raw ~ std_normal(); 
  
  // likelihood 
  for (n in 1:N) {
    RT[n] ~ RDM(a[subj[n]], v_chosen[n], v_unchosen[n,]);
  }
  
}

generated quantities {
  vector[N] log_lik; // log pointwise predictive density
  //matrix[N, 2] ppd; // prediction [choice, RT]
  real dev; 
  dev = 0;
  
  for (n in 1:N) {
    log_lik[n] = RDM_lpdf(RT[n] | a[subj[n]], v_chosen[n], v_unchosen[n,]); 
   // ppd[n,] = RDM_rng(a[subj[n]], v_chosen[n], v_unchosen[n,]);
    dev = dev - 2*log_lik[n];
  }
}

