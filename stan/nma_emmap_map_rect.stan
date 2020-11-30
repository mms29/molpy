functions{
    real gaussian_pdf(row_vector x, row_vector mu, real sigma){
        return pow((1/((2*pi()*square(sigma)))),(3.0/2.0))*exp(-((1/(2*square(sigma))) * square(distance(x,mu))));
    }
    vector lp_reduce(vector beta, vector theta, real[] xr, int[] xi){
        int n_atoms         = xi[1];
        int N               = xi[2];
        int halfN           = xi[3];
        int n               = xi[4];
        real epsilon        = xr[n*N*N+1];
        real gaussian_sigma = xr[n*N*N+2];
        real sampling_rate  = xr[n*N*N+3];

        matrix[n_atoms,3] x = to_matrix(beta, n_atoms, 3);

        real total=0;

        for (i in 1:n){
            for (j in 1:N){
                for (k in 1:N){
                    real s=0;
                    for(a in 1:n_atoms){
                        s += gaussian_pdf(x[a], ([i-halfN-1,j-halfN-1,k-halfN-1])*sampling_rate, gaussian_sigma);
                    }
                    total+= normal_lpdf(xr[k + (j-1)*N + (i-1)*N*N]| s, epsilon);
                }
            }
        }
        return [total]';
    }
}
data {

    // initial structure
    int<lower=0> n_atoms;
    matrix [n_atoms, 3] x0;
    int n_shards;

    // normal modes
    int<lower=0> n_modes;
    matrix [n_modes, 3] A[n_atoms];

    // em density
    int<lower=0> N;
    real em_density[N*N*N];

    // hyperparmeters
    real sigma;
    real epsilon;
    real mu;

    real sampling_rate;
    real gaussian_sigma;
    int halfN;
}
transformed data {

    int n = N/n_shards;
    real xr [n_shards, n*N*N + 3];
    int xi [n_shards, 4] ;
    vector [0]theta [n_shards];

    for(shard in 1:n_shards){
        xr[shard, 1:n*N*N] = em_density[(shard-1)*n*N*N + 1 :shard*n*N*N];
        xr[shard, n*N*N+1] =epsilon;
        xr[shard, n*N*N+2] =gaussian_sigma;
        xr[shard, n*N*N+3] =sampling_rate;
        xi[shard] = {n_atoms, N, halfN, n};
    }
}
parameters {
    row_vector<lower=-200,upper=200> [n_modes]q;
}
transformed parameters {
    matrix <lower=-halfN*sampling_rate,upper=halfN*sampling_rate> [n_atoms, 3] x;
    for (i in 1:n_atoms){
        x[i] = q*A[i] + x0[i];
    }

}
model {
    for (i in 1:n_modes){
        q[i] ~ normal(mu, sigma);
    }
    target += sum( map_rect( lp_reduce , to_vector(x) , theta , xr , xi ) );
}