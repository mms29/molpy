functions{
    real gaussian_pdf(row_vector x, row_vector mu, real sigma){
        return pow((1/((2*pi()*square(sigma)))),1.5)*exp(-((1/(2*square(sigma))) * square(distance(x,mu))));
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
    int<lower=0> dimX;
    int<lower=0> dimY;
    int<lower=0> dimZ;
    real em_density[dimX, dimY, dimZ];

    // hyperparmeters
    real sigma;
    real epsilon;
    real mu;

    real sampling_rate;
    real guassian_range;
    real gaussian_sigma;
    row_vector [3] center_transform;
}
parameters {
    row_vector<lower=-300,upper=300> [n_modes]q;
}
transformed parameters {
    matrix [n_atoms, 3] x;
    real sim_density[dimX, dimY, dimZ] = rep_array(0.0, dimX, dimY, dimZ);

    for (a in 1:n_atoms){
        row_vector pos = int((x[a]/sampling_rate) + center_transform);

        x[a] = q*A[a] + x0[a];
        for (i in (pos[1] - guassian_range -1) : (pos[1] + guassian_range)){
            if (i>0 and i <=dimX){
                for (j in (pos[2] - guassian_range -1) : (pos[2] + guassian_range)){
                    if(j>0 and j <=dimY){
                        for (k in (pos[3] - guassian_range -1) : (pos[3] + guassian_range)){
                            if(k>0 and k <=dimZ){
                                row_vector mu= {i,j,k}
                                sim_density+=gaussian_pdf(x[a],(mu-center_transform)*sampling_rate, gaussian_sigma);
                            }
                        }
                    }
                }
            }
        }
    }

}
model {
    for (i in 1:n_modes){
        q[i] ~ normal(mu, sigma);
    }
    em_density ~ normal(sim_density, epsilon);

}