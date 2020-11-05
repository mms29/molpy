
data {

    // initial structure
    int<lower=0> n_atoms;
    matrix [n_atoms, 3] x0;

    // normal modes
    int<lower=0> n_modes;
    matrix [n_modes, 3] A[n_atoms];

    // em density
    int<lower=0> dimX;
    int<lower=0> dimY;
    int<lower=0> dimZ;
    real em_density[dimX, dimY, dimZ];

    // hyperparmeters
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
    for (i in 1:dimX){
        for (j in 1:dimY){
            for (k in 1:dimZ){
                real s=0;
                for (a in 1:n_atoms){
                    s+= exp( -(square(i -(dimX/2.0) - x[a,1]) + square(j -(dimY/2.0) - x[a,2]) + square(k -(dimZ/2.0) - x[a,3])));
                }
                em_density[i,j,k] ~ normal(s, epsilon);
            }
        }
    }
}