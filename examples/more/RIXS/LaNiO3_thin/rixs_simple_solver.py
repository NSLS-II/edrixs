#!/usr/bin/env python

import edrixs

if __name__ == "__main__":
    F2_dd, F4_dd = 12.234 * 0.65, 7.598 * 0.65
    F0_dd = edrixs.get_F0('d', F2_dd, F4_dd)

    G1_dp, G3_dp = 5.787 * 0.7, 3.291 * 0.7
    F0_dp = edrixs.get_F0('dp', G1_dp, G3_dp)
    F2_dp = 7.721 * 0.95

    slater = ([F0_dd, F2_dd, F4_dd], [F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp])

    zeta_d_i = 0.083
    zeta_d_n = 0.102
    zeta_p_n = 11.24

    dq10, d1, d3 = 1.3, 0.05, 0.2
    cf = edrixs.cf_tetragonal_d(dq10, d1, d3)

    eval_i, eval_n, T_abs = edrixs.ed_1v1c(v_name='d',
                                           c_name='p',
                                           v_soc=(zeta_d_i, zeta_d_n),
                                           c_soc=zeta_p_n,
                                           v_noccu=8,
                                           slater=slater,
                                           cf_mat=cf)

    edrixs.write_tensor(eval_i, 'eval_i.dat')
    edrixs.write_tensor(eval_n, 'eval_n.dat')
