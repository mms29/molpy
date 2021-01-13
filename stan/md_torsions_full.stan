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
    real first_max;
    real torsion_max;

    real angle_sigma;
    real angle_max;
    real bond_sigma;
    real bond_max;

    real epsilon;
    real first_sigma;
    real torsion_sigma;

}
parameters {
    matrix<lower=-first_max,upper=first_max> [3,3] first_var;
    vector<lower=-torsion_max,upper=torsion_max> [n_atoms-3] torsion_var;
    vector<lower=-bond_max,upper=bond_max> [n_atoms-3] bond_var;
    vector<lower=-angle_max,upper=angle_max> [n_atoms-3] angle_var;
}
transformed parameters {
    matrix [n_atoms, 3] x;

    x[:3] =first + first_var;

    for (i in 4:n_atoms){
        row_vector [3] A = x[i-3];
        row_vector [3] B = x[i-2];
        row_vector [3] C = x[i-1];
        row_vector [3] AB = x[i-2]-x[i-3];
        row_vector [3] BC = x[i-1]-x[i-2];
        row_vector [3] bc = BC ./norm(BC);
        row_vector [3] n = cross_product(AB, bc) ./ norm(cross_product(AB, bc));

        matrix [3,3] M1 = generate_rotation_matrix(angles[i-3] + angle_var[i-3], n);
        matrix [3,3] M2 = generate_rotation_matrix(torsions[i-3]+torsion_var[i-3], bc);

        x[i] = x[i-1] + (bonds[i-3] +bond_var[i-3])*bc* M1' * M2';
    }
}
model {
    for (i in 1:n_atoms){
        y[i] ~ normal(x[i], epsilon);
    }
    for(i in 1:3){
        for( j in 1:3){
            first_var[i,j] ~ normal(0, first_sigma);
        }
    }
    torsion_var ~ normal(0, torsion_sigma);
    angle_var ~ normal(0, angle_sigma);
    bond_var ~ normal(0, bond_sigma);

}