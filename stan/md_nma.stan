
data {
    int<lower=0> n_atoms;
    int<lower=0> n_modes;
    matrix [n_atoms, 3] y;
    matrix [n_atoms, 3] x0;
    matrix [n_modes, 3] A[n_atoms];
    real sigma;
    real epsilon;
    real mu;

    real k_md;
    real U_init;
    real r_md;
    real s_md;
}
parameters {
    row_vector [n_modes]q;
    matrix [n_atoms, 3] x_md;
}
transformed parameters {
    matrix [n_atoms, 3] x;
    real U=0;
    for (i in 1:n_atoms){
        x[i] = q*A[i] + x0[i]+ x_md[i];
    }
    for (i in 1:n_atoms-1){
        U += k_md*square(sqrt( square(x[i,1]-x[i+1,1]) + square(x[i,2]-x[i+1,2]) + square(x[i,3]-x[i+1,3])) - r_md);
    }

}
model {
    for (i in 1:n_modes){
        q[i] ~ normal(mu, sigma);
    }
    for (i in 1:n_atoms){
        y[i] ~ normal(x[i], epsilon);
        x_md[i] ~ normal(0, s_md);
    }
    U ~ exponential(U_init);
}