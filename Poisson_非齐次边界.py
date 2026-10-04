import numpy as np
'''a:下对角线、b:主对角线、c：上对角线'''
def thomas(a, b, c, d):
    n = len(b)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i-1] * cp[i-1]
        if i < n - 1:
            cp[i] = c[i] / m
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / m
    x[n-1] = dp[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x
'''
基函数个数等于节点个数，并且满足公式：
∑(j=1→N-1) uⱼ ∫ ϕⱼ' ϕᵢ' dx = ∫ f ϕᵢ dx - u₀ ∫ ϕ₀' ϕᵢ' dx - u_N ∫ ϕ_N' ϕᵢ' dx

每一个基函数由三部分组成(除边界外)：
ϕ_j =(x - x_{j-1})/h  x_j-1 ~ x_j
     (x_j+1 - x)  /h  x_j ~ x_j+1
     0                else
'''
def assemble_poisson_1d(N):
    h = 1.0 / N
    n_unknow = N - 1
    F = np.zeros(n_unknow)
    K = np.zeros((n_unknow,n_unknow))
    x = np.linspace(0, 1, N + 1)

    for e in range(1,N):
        Fe = 0
        Fe_0 = -h * (3 * x[e] + h)
        Fe_1 = -h * (3 * x[e] - h)
        Fe += Fe_0
        Fe += Fe_1
        F[e-1] += Fe
    u0 = 1.0
    uN = 4.0
    F[0] += u0 / h
    F[-1] += uN / h
    Ke = (1.0 / h) * np.array([[1.0,-1.0],
                              [-1.0,1.0]])

    for e in range(N):
        nodes = [e, e+1]
        for a in range(2):
            I = nodes[a]
            if I == 0 or I == N:
                continue
            ia = I - 1
            for b in range(2):
                Ib = nodes[b]
                if Ib == 0 or Ib == N:
                    continue
                ib = Ib - 1
                K[ia,ib] += Ke[a,b]
    return F, K ,h

def solve_fem(N):
    F, K, h = assemble_poisson_1d(N)

    a = np.diag(K).copy()
    b = np.diag(K,1).copy()
    c = np.diag(K,-1).copy()

    U = thomas(c, a, b, F)

    x = np.linspace(0, 1, N + 1)
    u_h = np.zeros(N+1)
    u_h[0]  = 1.0
    u_h[-1] = 4.0
    u_h[1:N]  = U
    return x, u_h , h

""" 只针对基函数是P1的高斯求积 """
def l2_error_gauss(x, u_h, N):
    h = 1.0 / N         #  '''固定'''
    xi_g = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])      #  '''固定'''
    w_g  = np.array([5/9, 8/9, 5/9])                         #  '''固定'''

    integral_val = 0.0                                       #  '''固定'''

    for i in range(N):
        x_left = x[i]

        for q in range(3):
            xi = xi_g[q]
            w  = w_g[q]

            xq = x_left + 0.5 * h * (1.0 + xi)

            u_ex = xq**3 + 2.0 * xq + 1.0           # 基函数是P1时，更换函数只需要改这个

            phi0 = 0.5 * (1.0 - xi)
            phi1 = 0.5 * (1.0 + xi)
            u_h_val = phi0 * u_h[i] + phi1 * u_h[i+1]

            integral_val += 0.5 * h * w * (u_ex - u_h_val) ** 2

    return np.sqrt(integral_val)

if __name__ == "__main__":
    print(f"{'N':>5} {'max_u_h':>12} {'L2 error':>14} {'order':>8}")
    print("-" * 45)
    error = []
    NS = [10 , 20 ,40 ,80 ,160]

    for N in NS:
        x, u_h, h = solve_fem(N)
        err = l2_error_gauss(x, u_h, N)
        error.append(err)
        if len(error) == 1:
             order_str = "-"
        else:
            order = np.log(error[-2]/error[-1])/np.log(2.0)
            order_str = f"{order:.4f}"

        print(f"{N:>5d} {np.max(u_h):>12.6f} {err:>14.6e} {order_str:>8}")
