
functions{
    vector lp_reduce(vector beta, vector theta, real[] xr, int[] xi){
        int n_atoms = xi[1];
        int n_modes = xi[2];
        matrix [n_atoms,3] y = to_matrix(xr[1:(n_atoms*3)], n_atoms, 3);
        matrix [n_atoms,3] x0 = to_matrix(xr[(n_atoms*3 + 1):(n_atoms*3*2)], n_atoms, 3);
        matrix [n_modes, 3] A[n_atoms];
        row_vector[n_modes] q = real mu;(beta);
        real epsilon = xr[n_atoms*3*2 + n_modes*n_atoms*3 + 1];
        real s=0;

        for( a in 1:n_atoms){
            A[a] = to_matrix (xr[n_atoms*3*2 + (a-1)*n_modes*3 + 1: n_atoms*3*2 + a*n_modes*3], n_modes, 3 );
        }

        for (i in 1:n_atoms){
            s += normal_lpdf(y[i] | q*A[i] + x0[i], epsilon);
        }
        return [s]';
    }
}
data {
    int<lower=0> n_shards;
    int<lower=0> n_atoms;
    int<lower=0> n_modes;
    matrix [n_atoms, 3] y;
    matrix [n_atoms, 3] x0;
    matrix [n_modes, 3] A[n_atoms];
    real sigma;
    real epsilon;
    real mu;
}
transformed data {
    int n_aps = n_atoms / n_shards;
    int xi[n_shards, 2];
    real xr[n_shards, n_aps*3*2 + n_modes*n_aps*3 + 1];
    vector[0] theta[n_shards];

    for (i in 1:n_shards){
        xi[i] = {n_aps, n_modes};
        xr[i, 1:(n_aps*3)] = to_array_1d(y[(i-1)*n_aps +1:i*n_aps]);
        xr[i, (n_aps*3 + 1):(n_aps*3*2)] = to_array_1d(x0[(i-1)*n_aps +1:i*n_aps]);
        for( a in 1:n_aps){
            xr[i, (n_aps*3*2) + (a-1)*n_modes*3 + 1 : (n_aps*3*2) + a*n_modes*3] =to_array_1d(A[(i-1)*n_aps + a]);
        }
        xr[i, n_aps*3*2 + n_modes*n_aps*3 + 1] = epsilon;
    }
}
parameters {
    vector<lower=-300,upper=300> [n_modes]q;
}
model {
    for (i in 1:n_modes){
        q[i] ~ normal(mu, sigma);
    }
    target += sum( map_rect( lp_reduce , q , theta , xr , xi ) );
}