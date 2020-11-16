functions{
    real gaussian_pdf(row_vector x, row_vector mu, real sigma){
        return pow((1/((2*pi()*square(sigma)))),(3.0/2.0))*exp(-((1/(2*square(sigma))) * square(distance(x,mu))));
    }
    vector lp_reduce(vector beta, vector theta, real[] xr, int[] xi){
        int n_atoms         = xi[1];
        int dX              = xi[2];
        int dimY            = xi[3];
        int dimZ            = xi[4];

        real ct1            = xr[dX*dimY*dimZ+1];
        real ct2            = xr[dX*dimY*dimZ+2];
        real ct3            = xr[dX*dimY*dimZ+3];
        real epsilon        = xr[dX*dimY*dimZ+4];
        real gaussian_sigma = xr[dX*dimY*dimZ+5];
        real sampling_rate  = xr[dX*dimY*dimZ+6];

        matrix [n_atoms,3] x = to_matrix(beta, n_atoms, 3);

        real total=0;

        for(i in 1:dX){
            for (j in 1:dimY){
                for (k in 1:dimZ){
                    real s=0;
                    for (a in 1:n_atoms){
                        s+= gaussian_pdf(x[a], [(i-ct1)*sampling_rate,
                                                (j-ct2)*sampling_rate,
                                                (k-ct3)*sampling_rate], gaussian_sigma);
                    }
                    total += normal_lpdf(xr[ k + (j-1) * dimZ + (i-1) * dimY * dimZ] | s, epsilon);
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
    int<lower=0> dimX;
    int<lower=0> dimY;
    int<lower=0> dimZ;
    real em_density[dimX*dimY*dimZ];

    // hyperparmeters
    real sigma;
    real epsilon;
    real mu;

    real sampling_rate;
    real gaussian_sigma;
    vector [3] center_transform;
}
transformed data {

    int dX = dimX/n_shards;
    real xr [n_shards, dX*dimY*dimZ + 6];
    int xi [n_shards, 4] ;
    vector [0]theta [n_shards];

    for(shard in 1:n_shards){
        xr[shard, 1 : dX*dimY*dimZ] = em_density[(shard-1)*dX*dimY*dimZ + 1 :shard*dX*dimY*dimZ];
        xr[shard, dX*dimY*dimZ+1] =center_transform[1];
        xr[shard, dX*dimY*dimZ+2] =center_transform[2];
        xr[shard, dX*dimY*dimZ+3] =center_transform[3];
        xr[shard, dX*dimY*dimZ+4] =epsilon;
        xr[shard, dX*dimY*dimZ+5] =gaussian_sigma;
        xr[shard, dX*dimY*dimZ+6] =sampling_rate;
        xi[shard] = {n_atoms, dX, dimY, dimZ};
    }
}
parameters {
    row_vector<lower=-200,upper=200> [n_modes]q;
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
    target += sum( map_rect( lp_reduce , to_vector(x) , theta , xr , xi ) );
//    target += sum( lp_reduce(to_vector(x) , theta , xr , xi ) );


}