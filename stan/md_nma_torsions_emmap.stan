functions{
    row_vector cross_product(row_vector u, row_vector v){
        row_vector [3] r = [u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]];
        return r;
    }

    real norm(row_vector v){
        return sqrt(dot_self(v));
    }

    matrix generate_rotation_matrix(real angle, row_vector v){
        real ux = v[1];
        real uy = v[2];
        real uz = v[3];
        real c = cos(angle);
        real s = sin(angle);
        matrix [3,3] M= [[ ux*ux*(1-c) + c   , ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                         [ ux*uy*(1-c) + uz*s, uy*uy*(1-c) + c   , uy*uz*(1-c) - ux*s],
                         [ ux*uz*(1-c) - uy*s, uy*uz*(1-c) + ux*s, uz*uz*(1-c) + c   ]];
        return M;
    }

    real gaussian_pdf(matrix x, matrix y, real sigma){
        return sum(exp(-(square(x[:,1] -y[:,1]) + square(x[:,2] -y[:,2]) +square(x[:,3] -y[:,3]))/(2*square(sigma))));
    }
}
data {
    int<lower=0> n_atoms;

    real bonds [n_atoms-3];
    real angles [n_atoms-3];
    real torsions [n_atoms-3];
    matrix [3,3] first;

    real epsilon;
    real first_sigma;
    real torsion_sigma;
    real first_max;
    real torsion_max;

    // em density
    int<lower=0> N;
    real em_density[N, N, N];

    real sampling_rate;
    real gaussian_sigma;
    int halfN;

    // normal modes
    int<lower=0> n_modes;
    matrix [n_modes, 3] A[n_atoms];
    real q_max;
    real q_mu;
    real q_sigma;
}
parameters {
    matrix<lower=-first_max,upper=first_max>    [3,3] first_var;
    vector<lower=-torsion_max,upper=torsion_max> [n_atoms-3] torsion_var;
    row_vector<lower=-q_max,upper=q_max> [n_modes]q;
}
transformed parameters {
    matrix<lower=-halfN*sampling_rate,upper=halfN*sampling_rate> [n_atoms, 3] x;

    x[:3] =first + first_var;

    for (i in 4:n_atoms){
        row_vector [3] AB = x[i-2]-x[i-3];
        row_vector [3] BC = x[i-1]-x[i-2];
        row_vector [3] bc = BC ./norm(BC);
        row_vector [3] n = cross_product(AB, bc) ./ norm(cross_product(AB, bc));

        matrix [3,3] M1 = generate_rotation_matrix(angles[i-3], n);
        matrix [3,3] M2 = generate_rotation_matrix(torsions[i-3]+torsion_var[i-3], bc);

        x[i] = x[i-1] + bonds[i-3]*bc* M1' * M2';
    }

    for (i in 1:n_atoms){
        x[i] += q*A[i];
    }
}
model {
    for(i in 1:3){
        for( j in 1:3){
            first_var[i,j] ~ normal(0, first_sigma);
        }
    }
    q ~ normal(q_mu, q_sigma);
    torsion_var ~ normal(0, torsion_sigma);
    for (i in 1:N){
        for (j in 1:N){
            for (k in 1:N){
                target += normal_lpdf(em_density[i,j,k] | gaussian_pdf(x, rep_matrix([i-halfN-1,j-halfN-1,k-halfN-1]*sampling_rate, n_atoms), gaussian_sigma), epsilon);
            }
        }
    }
}