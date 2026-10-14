"""Disk-cached operator kernels, imported only when Numba is requested."""

import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def comb_jit(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = (result * (n - i)) // (i + 1)
    return result


@njit(cache=True, inline='always')
def popcount_u64(value):
    count = 0
    while value != np.uint64(0):
        value &= value - np.uint64(1)
        count += 1
    return count


@njit(cache=True, inline='always')
def hash_decoder_jit(rank, norb, nocc):
    state = np.uint64(0)
    j = nocc
    i = norb - 1
    c = comb_jit(i, j)
    while i >= 0 and j > 0:
        if c <= rank:
            state |= np.uint64(1) << np.uint64(norb - 1 - i)
            rank -= c
            old_i, old_j = i, j
            i -= 1
            j -= 1
            if j == 0 or i < 0:
                break
            c = (c * old_j) // old_i
        else:
            if i == 0:
                break
            c = (c * (i - j)) // i
            i -= 1
    return state


@njit(cache=True, inline='always')
def hash_encoder_jit(state, norb):
    if norb < 64:
        state &= (np.uint64(1) << np.uint64(norb)) - np.uint64(1)
    rank = 0
    k = 1
    for pos in range(norb - 1, -1, -1):
        if state & (np.uint64(1) << np.uint64(pos)):
            rank += comb_jit(norb - 1 - pos, k)
            k += 1
    return rank


@njit(cache=True, inline='always')
def sign_count_u64(state, orbital):
    prefix = state & ((np.uint64(1) << np.uint64(orbital)) - np.uint64(1))
    return 1 if popcount_u64(prefix) % 2 == 0 else -1


@njit(cache=True, inline='always')
def decode_combinadic_jit(index, norbs, noccus, offsets, sizes):
    state = np.uint64(0)
    for n in range(len(norbs)):
        size = sizes[n]
        shell_rank = index % size
        index //= size
        colex_rank = size - 1 - shell_rank
        shell_state = hash_decoder_jit(colex_rank, norbs[n], noccus[n])
        state |= shell_state << np.uint64(offsets[n])
    return state


@njit(cache=True, inline='always')
def encode_combinadic_jit(state, norbs, noccus, offsets, sizes, strides):
    index = 0
    for n in range(len(norbs)):
        norb = norbs[n]
        if norb == 64:
            mask = np.uint64(0xffffffffffffffff)
        else:
            mask = (np.uint64(1) << np.uint64(norb)) - np.uint64(1)
        shell_state = (state >> np.uint64(offsets[n])) & mask
        if popcount_u64(shell_state) != noccus[n]:
            return -1
        colex_rank = hash_encoder_jit(shell_state, norb)
        shell_rank = sizes[n] - 1 - colex_rank
        index += shell_rank * strides[n]
    return index


@njit(cache=True)
def build_combinadic(
    rb_norbs, rb_noccus, rb_offsets, rb_sizes, rb_strides,
    lb_norbs, lb_noccus, lb_offsets, lb_sizes, lb_strides,
    e_terms, e_vals, u_terms, u_vals, cstart, cend,
):
    rows = []
    cols = []
    vals = []

    for column in range(cstart, cend):
        state0 = decode_combinadic_jit(
            column, rb_norbs, rb_noccus, rb_offsets, rb_sizes
        )

        for t in range(e_terms.shape[0]):
            iorb = int(e_terms[t, 0])
            jorb = int(e_terms[t, 1])
            bit_j = np.uint64(1) << np.uint64(jorb)
            if state0 & bit_j == 0:
                continue
            sign_1 = sign_count_u64(state0, jorb)
            state = state0 ^ bit_j

            bit_i = np.uint64(1) << np.uint64(iorb)
            if state & bit_i != 0:
                continue
            sign_2 = sign_count_u64(state, iorb)
            state |= bit_i

            row = encode_combinadic_jit(
                state, lb_norbs, lb_noccus, lb_offsets, lb_sizes, lb_strides
            )
            if row != -1:
                rows.append(row)
                cols.append(column)
                vals.append(e_vals[t] * sign_1 * sign_2)

        for t in range(u_terms.shape[0]):
            lorb = int(u_terms[t, 0])
            korb = int(u_terms[t, 1])
            jorb = int(u_terms[t, 2])
            iorb = int(u_terms[t, 3])
            if iorb == jorb or korb == lorb:
                continue

            bit_i = np.uint64(1) << np.uint64(iorb)
            if state0 & bit_i == 0:
                continue
            sign_1 = sign_count_u64(state0, iorb)
            state = state0 ^ bit_i

            bit_j = np.uint64(1) << np.uint64(jorb)
            if state & bit_j == 0:
                continue
            sign_2 = sign_count_u64(state, jorb)
            state ^= bit_j

            bit_k = np.uint64(1) << np.uint64(korb)
            if state & bit_k != 0:
                continue
            sign_3 = sign_count_u64(state, korb)
            state |= bit_k

            bit_l = np.uint64(1) << np.uint64(lorb)
            if state & bit_l != 0:
                continue
            sign_4 = sign_count_u64(state, lorb)
            state |= bit_l

            row = encode_combinadic_jit(
                state, lb_norbs, lb_noccus, lb_offsets, lb_sizes, lb_strides
            )
            if row != -1:
                rows.append(row)
                cols.append(column)
                vals.append(u_vals[t] * sign_1 * sign_2 * sign_3 * sign_4)

    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(vals, dtype=np.complex128),
    )


@njit(cache=True)
def build_explicit(
    rb_states, lb_lookup,
    e_terms, e_vals, u_terms, u_vals, cstart, cend,
):
    rows = []
    cols = []
    vals = []

    for column in range(cstart, cend):
        state0 = rb_states[column]

        for t in range(e_terms.shape[0]):
            iorb = int(e_terms[t, 0])
            jorb = int(e_terms[t, 1])
            bit_j = np.uint64(1) << np.uint64(jorb)
            if state0 & bit_j == 0:
                continue
            sign_1 = sign_count_u64(state0, jorb)
            state = state0 ^ bit_j

            bit_i = np.uint64(1) << np.uint64(iorb)
            if state & bit_i != 0:
                continue
            sign_2 = sign_count_u64(state, iorb)
            state |= bit_i

            if state in lb_lookup:
                row = lb_lookup[state]
                rows.append(row)
                cols.append(column)
                vals.append(e_vals[t] * sign_1 * sign_2)

        for t in range(u_terms.shape[0]):
            lorb = int(u_terms[t, 0])
            korb = int(u_terms[t, 1])
            jorb = int(u_terms[t, 2])
            iorb = int(u_terms[t, 3])
            if iorb == jorb or korb == lorb:
                continue

            bit_i = np.uint64(1) << np.uint64(iorb)
            if state0 & bit_i == 0:
                continue
            sign_1 = sign_count_u64(state0, iorb)
            state = state0 ^ bit_i

            bit_j = np.uint64(1) << np.uint64(jorb)
            if state & bit_j == 0:
                continue
            sign_2 = sign_count_u64(state, jorb)
            state ^= bit_j

            bit_k = np.uint64(1) << np.uint64(korb)
            if state & bit_k != 0:
                continue
            sign_3 = sign_count_u64(state, korb)
            state |= bit_k

            bit_l = np.uint64(1) << np.uint64(lorb)
            if state & bit_l != 0:
                continue
            sign_4 = sign_count_u64(state, lorb)
            state |= bit_l

            if state in lb_lookup:
                row = lb_lookup[state]
                rows.append(row)
                cols.append(column)
                vals.append(u_vals[t] * sign_1 * sign_2 * sign_3 * sign_4)

    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(vals, dtype=np.complex128),
    )
