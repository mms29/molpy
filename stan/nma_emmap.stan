functions{
    real gaussian_pdf(row_vector x, row_vector mu, real sigma){
        return pow((1/((2*pi()*square(sigma)))),(3.0/2.0))*exp(-((1/(2*square(sigma))) * square(distance(x,mu))));
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
    row_vector [n_modes] mu;

    real sampling_rate;
    real gaussian_sigma;
    int halfN;
}
parameters {
    row_vector<lower=-200,upper=200> [n_modes]q;
}
transformed parameters {
    matrix<lower=-halfN*sampling_rate,upper=halfN*sampling_rate> [n_atoms, 3] x;

    real sim_density[N, N, N] = rep_array(0.0, N, N, N);
    for (a in 1:n_atoms){
        x[a] = q*A[a] + x0[a];
        for (i in 1:N){
            for (j in 1:N){
                for (k in 1:N){
                    sim_density[i,j,k] += gaussian_pdf(x[a], ([i-halfN-1,j-halfN-1,k-halfN-1])*sampling_rate, gaussian_sigma);
                }
            }
        }
    }
}
model {
    q ~ normal(mu, sigma);

    for (i in 1:N){
        for (j in 1:N){
            for (k in 1:N){
                em_density[i,j,k] ~ normal(sim_density[i,j,k], epsilon);
            }
        }
    }
}