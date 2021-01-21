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

}
data {
    int<lower=0> n_atoms;

    matrix [n_atoms, 3] y;

    real bonds [n_atoms-3];
    real angles [n_atoms-3];
    real torsions [n_atoms-3];
    matrix [3,3] first;

    real epsilon;
    real R_sigma;
    real shift_sigma;
    real torsion_sigma;
    real max_shift;

    real k_torsions;
    real n_torsions;
    real delta_torsions;

    real k_U;

//    int<lower=0> n_modes;
//    matrix [n_modes, 3] A_modes[n_atoms];
//    real sigma;
    int verbose ;

}
parameters {
    vector[n_atoms-3] torsion_var;
//    row_vector [n_modes]q;
    real<lower=-pi(),upper=pi()> alpha;
    real<lower=-pi()/2.0,upper=pi()/2.0> beta;
    real<lower=-pi(),upper=pi()> gamma;
    row_vector <lower=-max_shift, upper=max_shift> [3] shift;

}
transformed parameters {
    matrix [n_atoms, 3] x;
    real U=0;
    matrix [3,3] R = [[cos(gamma) * cos(alpha) * cos(beta) - sin(gamma) * sin(alpha), cos(gamma) * cos(beta)*sin(alpha) + sin(gamma) * cos(alpha), -cos(gamma) * sin(beta)],
                   [-sin(gamma) * cos(alpha) * cos(beta) - cos(gamma) * sin(alpha),-sin(gamma) * cos(beta)*sin(alpha) + cos(gamma) * cos(alpha), sin(gamma) * sin(beta)],
                   [sin(beta)*cos(alpha), sin(beta)*sin(alpha), cos(beta)]];

    for (i in 1:3){
        x[i] = first[i]*R +shift;
    }

    for (i in 4:n_atoms){
        row_vector [3] A = x[i-3];
        row_vector [3] B = x[i-2];
        row_vector [3] C = x[i-1];
        row_vector [3] AB = B-A;
        row_vector [3] BC = C-B;
        row_vector [3] bc = BC ./norm(BC);
        row_vector [3] n = cross_product(AB, bc) ./ norm(cross_product(AB, bc));

        matrix [3,3] M1 = generate_rotation_matrix(angles[i-3], n);
        matrix [3,3] M2 = generate_rotation_matrix(torsion_var[i-3], bc);

        row_vector [3] D0 = bonds[i-3]*bc;
        row_vector [3] D1 = D0 * M1';
        row_vector [3] D2 = D1 * M2';

        x[i] = C + D2;

        U += k_torsions*(1 + cos(n_torsions*torsions[i-3] - delta_torsions));
    }

//    for (i in 1:n_atoms){
//        x[i] = q*A_modes[i] + x[i];
//    }
}
model {
    real likelihood = 0;
    real torsions_lp =0;
    real U_lp =0;

    for (i in 1:n_atoms){
        likelihood += normal_lpdf(y[i] | x[i], epsilon);
    }

    torsions_lp +=  normal_lpdf(torsion_var | torsions, torsion_sigma);

    U_lp += -k_U*U;

//    for (i in 1:n_modes){
//        q[i] ~ normal(0, sigma);
//    }

    alpha ~ normal(0,R_sigma);
    beta ~ normal(0,R_sigma/2);
    gamma ~ normal(0,R_sigma);
    shift ~ normal(0,shift_sigma);

    target += torsions_lp +U_lp + likelihood ;
    if (verbose){
        print("Likelihood=", likelihood);
        print("torsions=", torsions_lp);
        print("U=", U_lp);
        print(" ");
    }


}