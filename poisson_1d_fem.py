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

def assemble_poisson_1d(N):
    h = 1.0 / N
    n = N - 1

    F = np.zeros(n)
    K = np.zeros((n,n))

    Fe = (h / 2.0) * np.array([1.0,1.0])
    Ke = (1.0 / h) * np.array([[1.0,-1.0],
                              [-1.0,1.0]])

    for e in range(N):
        nodes = [e, e+1]
        for a in range(2):
            I = nodes[a]
            if I == 0 or I == N:
                continue
            ia = I - 1
            F[ia] += Fe[a]
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
    u_h[0]  = 0.0
    u_h[-1] = 0.0
    u_h[1:N]  = U
    return x, u_h , h


def l2_error_continuous(x, u_h, N):
    h = 1.0 / N  # 或者 h = x[1] - x[0]
    integral_val = 0.0

    for i in range(N):
        x_lef = x[i]
        u_exl = x_lef * (1 - x_lef) / 2
        err_l = (u_exl - u_h[i]) ** 2

        x_rig = x[i+1]
        u_exr = x_rig * (1 - x_rig) / 2
        err_r = (u_exr - u_h[i+1]) ** 2

        err = (err_l + err_r) / 2 * h
        integral_val += err
    # 4. 最后开根号
    return np.sqrt(integral_val)

if __name__ == "__main__":
    print(f"{'N':>5} {'max_u_h':>12} {'L2 error':>14} {'order':>8}")
    print("-" * 45)
    error = []
    NS = [10 , 20 ,40 ,80 ,160]

    for N in NS:
        x, u_h, h = solve_fem(N)
        err = l2_error_continuous(x, u_h, N)
        error.append(err)
        if len(error) == 1:
             order_str = "-"
        else:
            order = np.log(error[-2]/error[-1])/np.log(2.0)
            order_str = f"{order:.4f}"

        print(f"{N:>5d} {np.max(u_h):>12.6f} {err:>14.6e} {order_str:>8}")

