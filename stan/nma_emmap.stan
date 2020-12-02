functions{
    real gaussian_pdf(matrix x, matrix y, real sigma){
        return sum(exp(-(square(x[:,1] -y[:,1]) + square(x[:,2] -y[:,2]) +square(x[:,3] -y[:,3]))/(2*square(sigma))));
    }

}
data {

    // initial structure
    int<lower=0> n_atoms;
    matrix [n_atoms, 3] x0;

    // normal modes
    int<lower=0> n_modes;
    matrix [n_modes, 3] A[n_atoms];

    // em density
    int<lower=0> N;
    real em_density[N, N, N];

    // hyperparmeters
    real sigma;
    real epsilon;
    real mu;

    real sampling_rate;
    real gaussian_sigma;
    int halfN;
}
parameters {
    row_vector<lower=-200,upper=200> [n_modes]q;
}
transformed parameters {
    matrix<lower=-halfN*sampling_rate,upper=halfN*sampling_rate> [n_atoms, 3] x;
    for (a in 1:n_atoms){
        x[a] = q*A[a] + x0[a];
    }
}
model {
    q ~ normal(mu, sigma);

    for (i in 1:N){
        for (j in 1:N){
            for (k in 1:N){
                target += normal_lpdf(em_density[i,j,k] | gaussian_pdf(x, rep_matrix([i-halfN-1,j-halfN-1,k-halfN-1]*sampling_rate, n_atoms), gaussian_sigma), epsilon);
            }
        }
    }
}