
data {
    int<lower=0> n_atoms;
    int<lower=0> n_modes;
    matrix [n_atoms, 3] y;
    matrix [n_atoms, 3] x0;
    matrix [n_modes, 3] A[n_atoms];
    real sigma;
    real epsilon;
    row_vector [n_modes] mu;
}
parameters {
    row_vector<lower=-300,upper=300> [n_modes]q;
    real<lower=0,upper=2*pi()> alpha;
    real<lower=0,upper=pi()> beta;
    real<lower=0,upper=2*pi()> gamma;
}
transformed parameters {
    matrix [n_atoms, 3] x;
    matrix [3,3] R = [[cos(gamma) * cos(alpha) * cos(beta) - sin(gamma) * sin(alpha), cos(gamma) * cos(beta)*sin(alpha) + sin(gamma) * cos(alpha), -cos(gamma) * sin(beta)],
                   [-sin(gamma) * cos(alpha) * cos(beta) - cos(gamma) * sin(alpha),-sin(gamma) * cos(beta)*sin(alpha) + cos(gamma) * cos(alpha), sin(gamma) * sin(beta)],
                   [sin(beta)*cos(alpha), sin(beta)*sin(alpha), cos(beta)]]
    for (i in 1:n_atoms){
        x[i] = (q*A[i] + x0[i])*R;
    }

}
model {
    for (i in 1:n_modes){
        q[i] ~ normal(mu, sigma);
    }
    for (i in 1:n_atoms){
        y[i] ~ normal(x[i], epsilon);
    }
}