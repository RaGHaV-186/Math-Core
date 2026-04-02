import numpy as np


def swap_rows(M, row_index_1, row_index_2):
    M = M.copy()
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M


def get_index_first_non_zero_value_from_column(M, column, starting_row):
    column_array = M[starting_row:, column]
    for i, val in enumerate(column_array):
        if not np.isclose(val, 0, atol=1e-5):
            return i + starting_row
    return -1


def get_index_first_non_zero_value_from_row(M, row, augmented=False):
    M = M.copy()
    if augmented:
        M = M[:, :-1]
    row_array = M[row]
    for i, val in enumerate(row_array):
        if not np.isclose(val, 0, atol=1e-5):
            return i
    return -1


def augmented_matrix(A, B):
    return np.hstack((A, B))


def row_echelon_form(A, B):
    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):
        return 'Singular system'

    A = A.copy().astype('float64')
    B = B.copy().astype('float64')
    num_rows = len(A)
    M = augmented_matrix(A, B)

    for row in range(num_rows):
        pivot_candidate = M[row, row]

        if np.isclose(pivot_candidate, 0):
            first_non_zero_value_below_pivot_candidate = get_index_first_non_zero_value_from_column(M, row, row)
            M = swap_rows(M, row, first_non_zero_value_below_pivot_candidate)
            pivot = M[row, row]
        else:
            pivot = pivot_candidate

        M[row] = (1 / pivot) * M[row]

        for j in range(row + 1, num_rows):
            value_below_pivot = M[j, row]
            M[j] = M[j] - value_below_pivot * M[row]

    return M


def back_substitution(M):
    M = M.copy()
    num_rows = M.shape[0]

    for row in reversed(range(num_rows)):
        substitution_row = M[row]
        index = get_index_first_non_zero_value_from_row(M, row, augmented=True)

        for j in range(row):
            row_to_reduce = M[j]
            value = row_to_reduce[index]
            row_to_reduce = row_to_reduce - value * substitution_row
            M[j, :] = row_to_reduce

    return M[:, -1]


def gaussian_elimination(A, B):
    row_echelon_M = row_echelon_form(A, B)
    if isinstance(row_echelon_M, str):
        return row_echelon_M
    return back_substitution(row_echelon_M)


# Custom helper to replace the missing 'utils' function
def string_to_augmented_matrix(equations_str):
    import re
    lines = [line.strip() for line in equations_str.strip().split('\n')]
    # Identify variables (x, y, w, z etc)
    vars_found = sorted(list(set(re.findall(r'[a-z]', equations_str))))

    A_list = []
    B_list = []

    for line in lines:
        # Split equation into Left Hand Side and Right Hand Side
        lhs, rhs = line.split('=')
        B_list.append(float(rhs.strip()))

        coeffs = []
        for v in vars_found:
            # Find coefficient for each variable
            match = re.search(fr'([+-]?\s*\d*)\s*\*\s*{v}', lhs)
            if match:
                c_str = match.group(1).replace(' ', '')
                if c_str in ['', '+']:
                    coeffs.append(1.0)
                elif c_str == '-':
                    coeffs.append(-1.0)
                else:
                    coeffs.append(float(c_str))
            else:
                coeffs.append(0.0)
        A_list.append(coeffs)

    return " ".join(vars_found), np.array(A_list), np.array(B_list).reshape(-1, 1)


# --- Execution ---

equations = """
3*x + 6*y + 6*w + 8*z = 1
5*x + 3*y + 6*w = -10
4*y - 5*w + 8*z = 8
4*w + 8*z = 9
"""

variables, A_mat, B_mat = string_to_augmented_matrix(equations)
sols = gaussian_elimination(A_mat, B_mat)

if not isinstance(sols, str):
    for variable, solution in zip(variables.split(' '), sols):
        print(f"{variable} = {solution:.4f}")
else:
    print(sols)