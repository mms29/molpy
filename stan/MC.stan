functions{
    row_vector cross_product(row_vector u, row_vector v){
        row_vector [3] r = [u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]];
        return r;
    }

    real norm(row_vector v){
        return sqrt(dot_self(v));
    }
}
data {
    int<lower=0> n_atoms;
    matrix [n_atoms, 3] y;
    matrix [n_atoms, 3] x0;
    real epsilon;

    //potential energy def
    real nu;
    real k_U;

    // bonds
    real k_bonds;
    real r0_bonds;

    // angles
    real k_angles;
    real theta0_angles;

    // torsions
    real k_torsions;
    real delta_torsions;
    real n_torsions;

    real verbose;
}
parameters {
    matrix [n_atoms, 3] x_md;
}
transformed parameters {
    matrix [n_atoms, 3] x = x_md +x0;
    real U_bonds=0;
    real U_angles=0;
    real U_torsions=0;

    // potential energy
    for (i in 1:n_atoms){

        if (i< n_atoms){
            row_vector[3] u1=x[i+1]-x[i];
            real bonds = norm(u1);
            U_bonds+= k_bonds * square(bonds - r0_bonds);

            if (i < n_atoms - 1){
                row_vector[3] u2=x[i+2]-x[i+1];
                real angles = acos(dot_product(u1, u2) / (norm(u1) * norm(u2)));
                U_angles+= k_angles * square(angles - theta0_angles);

                if (i < n_atoms - 2){
                    row_vector[3] u3=x[i+3]-x[i+2];
                    real torsions = atan2(dot_product(norm(u2) * u1, cross_product(u2, u3)), dot_product(cross_product(u1,u2), cross_product(u2,u3)));
                    U_torsions += k_torsions*(1 + cos(n_torsions*torsions- delta_torsions));
                }
            }
        }

    }

}
model {
    real likelihood = 0;
    real x_md_lp = 0;
    real U_lp =0;

    for (i in 1:n_atoms){
        likelihood += normal_lpdf(y[i] | x[i], epsilon);
        x_md_lp += normal_lpdf(x_md[i] | 0 , nu);
    }

    U_lp = -k_U* (U_angles + U_torsions + U_bonds);

    target += x_md_lp +U_lp + likelihood ;
    if (verbose){
        print("Likelihood=", likelihood);
        print("x_md_lp=", x_md_lp);
        print("U=", U_lp);
        print("   U_bonds=", U_bonds);
        print("   U_angles=", U_angles);
        print("   U_torsions=", U_torsions);
        print(" ");
    }

}