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
}
parameters {
    row_vector<lower=-200,upper=200> [n_modes]q;
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
                real s=0;
                for(a in 1:n_atoms){
                    s += gaussian_pdf(x[a], ([i-halfN-1,j-halfN-1,k-halfN-1])*sampling_rate, gaussian_sigma);
                }
                target += normal_lpdf(em_density[i,j,k] | s, epsilon);
            }
        }
    }
    for (i in 1:n_atoms){
        x_md[i] ~ normal(0, s_md);
    }
    U ~ exponential(U_init);
}