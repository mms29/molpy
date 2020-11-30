
data {
    int<lower=0> n_atoms;
    int<lower=0> n_modes;
    matrix [n_atoms, 3] y;
    matrix [n_atoms, 3] x0;
    matrix [n_modes, 3] A[n_atoms];
    real sigma;
    real epsilon;
    real mu;

    //potential energy def
    real U_init;
    real s_md;

    // bonds
    real k_r;
    real r0;

    // angles
    real k_theta;
    real theta0;

    real k_lj;
    real d_lj;
}
parameters {
    row_vector [n_modes]q;
    matrix [n_atoms, 3] x_md;
    real<lower=-pi(),upper=pi()> alpha;
    real<lower=-pi()/2.0,upper=pi()/2.0> beta;
    real<lower=-pi(),upper=pi()> gamma;
}
transformed parameters {
    matrix [n_atoms, 3] x;
    real U=0;
    matrix [3,3] R = [[cos(gamma) * cos(alpha) * cos(beta) - sin(gamma) * sin(alpha), cos(gamma) * cos(beta)*sin(alpha) + sin(gamma) * cos(alpha), -cos(gamma) * sin(beta)],
                   [-sin(gamma) * cos(alpha) * cos(beta) - cos(gamma) * sin(alpha),-sin(gamma) * cos(beta)*sin(alpha) + cos(gamma) * cos(alpha), sin(gamma) * sin(beta)],
                   [sin(beta)*cos(alpha), sin(beta)*sin(alpha), cos(beta)]];
    // normal modes deformation
    for (i in 1:n_atoms){
        x[i] = (q*A[i] + x0[i]+ x_md[i])*R;
    }

    // potential energy
    for (i in 1:n_atoms){
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
    for (i in 1:n_modes){
        q[i] ~ normal(mu, sigma);
    }
    for (i in 1:n_atoms){
        y[i] ~ normal(x[i], epsilon);
        x_md[i] ~ normal(0, s_md);
    }
    alpha ~ normal(0,pi());
    beta ~ normal(0,pi()/2.0);
    gamma ~ normal(0,pi());
    U ~ exponential(U_init);
}