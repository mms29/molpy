functions{
    row_vector cross_product(row_vector u, row_vector v){
        row_vector [3] r = [u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]];
        return r;
    }

    real norm(row_vector v){
        return sqrt(dot_self(v));
    }

    real gaussian_pdf(matrix x, matrix y, real sigma){
        return sum(exp(-(square(x[:,1] -y[:,1]) + square(x[:,2] -y[:,2]) +square(x[:,3] -y[:,3]))/(2*square(sigma))));
    }
}
data {
    int<lower=0> n_atoms;
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

    // normal modes
    int<lower=0> n_modes;
    matrix [n_modes, 3] A_modes[n_atoms];
    real q_sigma;

    real verbose;

    int<lower=0> N;
    real density[N, N, N];
    real sampling_rate;
    real gaussian_sigma;
    int halfN;
}
parameters {
    matrix [n_atoms, 3] x_md;
    row_vector [n_modes]q_modes;
}
transformed parameters {
    matrix<lower=-halfN*sampling_rate,upper=halfN*sampling_rate> [n_atoms, 3] x;
    real U_bonds=0;
    real U_angles=0;
    real U_torsions=0;

    for (a in 1:n_atoms){
        x[a] = q_modes*A_modes[a] + x0[a] + x_md[a];
    }

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
    real modes_lp=0;

    for (i in 1:N){
        for (j in 1:N){
            for (k in 1:N){
                likelihood += normal_lpdf(density[i,j,k] | gaussian_pdf(x, rep_matrix([i-halfN-1,j-halfN-1,k-halfN-1]*sampling_rate, n_atoms), gaussian_sigma), epsilon);
            }
        }
    }
    for (i in 1:n_atoms){
        x_md_lp += normal_lpdf(x_md[i] | 0 , nu);
    }
    modes_lp += normal_lpdf(q_modes | 0, q_sigma);

    U_lp = -k_U* (U_angles + U_torsions + U_bonds);

    target += x_md_lp +U_lp + likelihood  + modes_lp;
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