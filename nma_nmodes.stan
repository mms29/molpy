
data {
    int<lower=0> n_atoms;
    int<lower=0> n_modes;
    matrix [n_atoms, 3] y;
    matrix [n_atoms, 3] x0;
    matrix [n_modes, 3] A[n_atoms];
    real sigma;
    real epsilon;
    real mu;
}
parameters {
    row_vector [n_modes]q;
}
transformed parameters {
    matrix [n_atoms, 3] x;
    for (i in 1:n_atoms){
        x[i] = q*A[i] + x0[i];
    }

}
model {
    for (i in 1:n_modes){
        q[i] ~ normal(mu, sigma);
    }
    for (i in 1:n_atoms){
        y[i] ~ normal(x[i], epsilon);
    }
}