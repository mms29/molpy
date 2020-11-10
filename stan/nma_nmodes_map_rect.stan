/*functions {
  real partial_sum(real[,] y, int start, int end, matrix x0, matrix[] A, row_vector q, real epsilon) {
    real s;
    for(i in start:end){
        s+= normal_lpdf(y[i] | q*A[i]+ x0[i], epsilon);
    }
    return s;
  }
}
data {
    int<lower=0> n_atoms;
    int<lower=0> n_modes;
    real y[n_atoms, 3];
    matrix [n_atoms, 3] x0;
    matrix [n_modes, 3] A[n_atoms];
    real sigma;
    real epsilon;
    real mu;
}
parameters {
    row_vector [n_modes]q;
}
model {
    int grainsize = 1;
    for (i in 1:n_modes){
        q[i] ~ normal(mu, sigma);
    }
    target += reduce_sum(partial_sum, y, grainsize x0, A, q, epsilon);
    //partial_sum(y, 1, n_atoms, x0, A, q, epsilon);
    //for(i in 1:n_atoms){target+= normal_lpdf(y[i] | q*A[i]+ x0[i], epsilon);}

}

*/
functions {
  vector lp_reduce( vector beta , vector theta , real[] xr , int[] xi ) {
    int n_atoms = xi[1];
    int n_modes = xi[2];
    matrix [n_atoms, 3] y = to_matrix(xr[1:(n_atoms*3)], n_atoms, 3);
    matrix [n_atoms, 3] x0 = to_matrix(xr[(n_atoms*3 + 1):(n_atoms*3*2)], n_atoms, 3);
    matrix [n_modes, 3] A[n_atoms];
    real epsilon = xr[n_atoms*3 + n_atoms*3 + n_atoms*n_modes*3+ 1];
    real s;
    row_vector[n_modes] q = to_row_vector(beta);

    for( a in 1:n_atoms){
        A[a] = to_matrix (xr[n_atoms*3*2 + (a-1)*n_modes*3 + 1: n_atoms*3*2 + a*n_modes*3], n_modes, 3 );
    }
    for(i in 1:n_atoms){
        s+= normal_lpdf(y[i] | q*A[i] + x0[i], epsilon);
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
  int<lower = 0> n_aps = n_atoms / n_shards;
  real xr[n_shards, n_aps*3 + n_aps*3 + n_aps*n_modes*3+ 1];
  int  xi[n_shards, 2];
  vector[0] theta[n_shards];
  for( i in 1:n_shards){
    int j = 1 + (i-1)*n_aps;
    int k = i*n_aps;

    xr[i , 1:(n_aps*3)] = to_array_1d(y[j:k]);
    xr[i , (n_aps*3 + 1):(n_aps*3*2)] = to_array_1d(x0[j:k]);
    for( a in 1:n_aps){
        xr[i ,(n_aps*3*2) + (a-1)*n_modes*3 + 1 : (n_aps*3*2) + a*n_modes*3] =to_array_1d(A[(i-1)*n_aps + a]);
    }
    xr[i , n_aps*3 + n_aps*3 + n_aps*n_modes*3+ 1] = epsilon;
    xi[i]= {n_aps, n_modes};
  }
}
parameters {
  vector[n_modes] q;
}
model {
  for (i in 1:n_modes){
        q[i] ~ normal(mu, sigma);
   }
  target += sum( map_rect( lp_reduce , q , theta , xr , xi ) );
}