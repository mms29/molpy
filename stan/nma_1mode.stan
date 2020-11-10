data {
    int<lower=0> n_atoms;
    matrix [n_atoms, 3] y;
    matrix [n_atoms, 3] x0;
    matrix [n_atoms,3] A;
    real sigma;
    real epsilon;
    real mu;
}
parameters {
    real q;
}
transformed parameters {
    matrix [n_atoms, 3] x;

    for (i in 1:n_atoms){
        x[i] = q*A[i] + x0[i];
    }

}
model {
    q ~ normal(mu, sigma);

    for (i in 1:n_atoms){
        y[i] ~ normal(x[i], epsilon);
    }
}