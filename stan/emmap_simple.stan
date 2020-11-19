data {
    int N;
    matrix [N,N] x_map;
    real sigma;
    real epsilon;
    row_vector [2] mu;
}
parameters {
    row_vector<lower=-N, upper=N*2> [2] x;
}
transformed parameters{
    matrix [N,N] x_map_sim;

    for(i in 1:N){
        for(j in 1:N){
            x_map_sim[i,j] = exp(-square(distance(x,[i-1,j-1])));
        }
    }

}
model{
    x ~ normal(mu,sigma);

    for(i in 1:N){
        for(j in 1:N){
            x_map[i,j] ~ normal(x_map_sim[i,j], epsilon);
        }
    }
}