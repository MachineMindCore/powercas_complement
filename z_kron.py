import numpy_CAS as np
from utils import show_variable
# Helper

def compose(M, M_down, M_right, m_ll):
    """
    Matriz composition:
    [
        [M, M_right],
        [M_down, m_ll]
    ]
    """
    zero_pivot = np.array([[0]], dtype=M.dtype)
    M_a = np.append(M, M_down, axis=0)
    M_b = np.append(M_right, zero_pivot, axis=0)
    M_new = np.append(M_a, M_b, axis=1)
    M_new[M_new.shape[0]-1, M_new.shape[1]-1] = m_ll
    return M_new 

def reduce_kron(M, connection):
    """
    Kron reduction
    """
    print("===========================================")
    print("Kron reduction")
    p = connection[0] - 1
    q = connection[1] - 1
    last = len(M) - 1
    M_old = M.delete(last, axis=0).delete(last, axis=1)

    if q != -1:
        dZ = (M.get_col(q) - M.get_col(p)).delete(last, axis=0)
    else:
        dZ = (M.get_col(p)*-1).delete(last, axis=0)
    M_reduced = M_old - np.matmul(dZ, dZ.transpose(), dtype=complex)*(1/M[last,last])
    show_variable("M_old", M_old)
    show_variable("dZ", dZ)
    show_variable("M_reduced", M_reduced)
    print("===========================================")
    return M_reduced

# Rules

def sature_dim(func):
    def wrapper(self, *args, **kwargs):
        actual_dim = len(self.Z) if self.Z!=None else 0
        if actual_dim >= self.dim:
            self.reduce()
        object_result = func(self, *args, **kwargs)            
        return object_result
    return wrapper

def rule_0(_, z_q0, connection):
    """
    Start Z to reference
    """
    print("Init Z: conn -> "+str(connection))
    print("Z_k+1 -> [[z_q0]")
    Z_new = np.array([[z_q0]], dtype=complex)
    print(Z_new)
    return Z_new


def rule_1(Z, z_pq, connection):
    """
    Line to existing (z_pq)
    """
    print("Rule 1 -> {conn}".format(conn=connection))
    print("Z_k+1 -> [[Z_k, Z_:p], [Z_p:, Z_pp+z_pq]]")
    p = connection[0] - 1
    q = connection[1] - 1
    Z_down = Z.get_row(p)
    Z_right = Z.get_col(p)
    z_ll = Z[p,p] + z_pq
    Z_new = compose(Z, Z_down, Z_right, z_ll)

    show_variable("Z_k+1", Z_new)
    return Z_new

def rule_2(Z, z_q0, connection):
    """
    New to reference (z_q0)
    """
    print("Rule 2 -> {conn}".format(conn=connection))
    print("Z_k+1 -> [[Z_k, 0], [0, z_q0]]")
    Z_down = np.zeros((1, len(Z)-1))
    Z_right = np.zeros((len(Z)-1, 1))
    Z_new = compose(Z, Z_down, Z_right, z_q0)

    show_variable("Z_k+1", Z_new)
    return Z_new

def rule_3(Z, z_pq, connection):
    """
    Existing to existing (z_pq)
    """
    print("Rule 3 -> {conn}".format(conn=connection))
    print("Z_k+1 -> [[Z_k, Z_q:-Z_p:], [Z_:q-Z_:p, z_pq+Z_pp+Z_qq-2Z_pq]]")
    p = connection[0] - 1
    q = connection[1] - 1
    Z_down = Z.get_row(q) - Z.get_row(p)
    Z_right = Z.get_col(q) - Z.get_col(p)
    z_ll = z_pq + Z[p,p] + Z[q,q] - 2*Z[p,q]
    Z_new = compose(Z, Z_down, Z_right, z_ll)

    show_variable("Z_k+1", Z_new)
    return Z_new

def rule_4(Z, z_p0, connection):
    """
    Existing to reference (z_p0)
    """
    print("Rule 4 -> {conn}".format(conn=connection))
    print("Z_k+1 -> [[Z_k, -Z_p:], [-Z_:p, z_p0+Z_pp]]")
    p = connection[0] - 1
    q = connection[1] - 1
    Z_down = Z.get_row(p)*-1
    Z_right = Z.get_col(p)*-1
    z_ll = Z[p,p] + z_p0
    Z_new = compose(Z, Z_down, Z_right, z_ll)

    show_variable("Z_k+1", Z_new)
    return Z_new


# Models

class GraphZ:
    def __init__(self, dim) -> None:
        self.Z = None
        self.dim = dim
        self.connection = None
        
    def define_rule(self, connection):
        p = connection[0] - 1
        q = connection[1] - 1
        Z_len = len(self.Z) if self.Z != None else 0

        if Z_len == 0:
            return 0
        
        elif q == -1:
            if p>Z_len:
                return 2
            else:
                return 4
        else:
            if p<Z_len and q>=Z_len:
                return 1
            else:
                return 3


    def reduce(self):
        self.Z = reduce_kron(self.Z, self.connection)
        return self

    @sature_dim
    def add(self, z_pq, connection):
        print("------------------------------------------------")
        self.connection = connection
        rule = self.define_rule(connection)
        self.Z = globals()["rule_"+str(rule)](self.Z, z_pq, connection)
        return self

        