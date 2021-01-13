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

    //potential energy def
    real U_init;
    real s_md;

    // bonds
    real k_r;
    real r0;

    // angles
    real k_theta;
    real theta0;

    // lennard jones
    real k_lj;
    real d_lj;

    real q_max;
}
parameters {
    row_vector<lower=-q_max,upper=q_max> [n_modes]q;
    matrix[n_atoms, 3] x_md;
}
transformed parameters {
    matrix<lower=-halfN*sampling_rate,upper=halfN*sampling_rate> [n_atoms, 3] x;
    real U=0;
    for (i in 1:n_atoms){
        x[i] = q*A[i] + x_md[i] + x0[i];
    }

    for (i in 1:n_atoms){
        // potential energy
        if (i<n_atoms){
            U += k_r*square(distance(x[i], x[i+1]) - r0);
        }
        if (i+1<n_atoms){
            U += k_theta*square(acos(dot_product(x[i]-x[i+1],x[i+1]-x[i+2])/(distance(x[i],x[i+1])*distance(x[i+1],x[i+2]))) - theta0);
        }
        for (j in 1:n_atoms){
            if(i!=j){
                U+= 4*k_lj*(pow(d_lj/distance(x[i], x[j]),12) - pow(d_lj/distance(x[i], x[j]),6));
            }
        }
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
    for (i in 1:n_atoms){
        x_md[i] ~ normal(0, s_md);
    }
    U ~ exponential(U_init);
}