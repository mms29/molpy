
data {
    int<lower=0> n_atoms_y;
    int<lower=0> n_atoms_x;
    int<lower=0> n_modes;
    matrix [n_atoms_y, 3] y;
    matrix [n_atoms_x, 3] x0;
    matrix [n_modes, 3] A[n_atoms_x];
    real sigma;
    real epsilon;
    real mu;
    real threshold;
}
parameters {
    row_vector<lower=-300,upper=300> [n_modes]q;
}
transformed parameters {
    matrix [n_atoms_x, 3] x;
    matrix [n_atoms_x, n_atoms_y] dist;
    for (i in 1:n_atoms_x){
        x[i] = q*A[i] + x0[i];
        for (j in 1:n_atoms_y){
            dist[i,j] = distance(x[i], y[j]);
        }
    }

}
model {
    for (i in 1:n_modes){
        q[i] ~ normal(mu, sigma);
    }
    for (i in 1:n_atoms_x){
        if (min(dist[i])< threshold){
            dist[i] ~ exponential(epsilon);
        }
    }
}