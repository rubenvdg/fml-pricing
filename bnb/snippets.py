    # def new_upper_bound_dual(self, cube):

    #     E, k = self.problem.E, self.problem.k
    #     n, m = self.n, self.m

    #     # x_delta, _ = self._get_x_r_delta(cube)

    #     x_ub = cube.center + cube.radius
    #     x_lb = cube.center - cube.radius
    #     # x_ub = x_delta
    #     # x_lb = x_delta

    #     z_lb = np.exp(-self.problem.p_ub * self.problem.b)
    #     z_ub = np.exp(-self.problem.p_lb * self.problem.b)

    #     kx_ub = k @ x_ub

    #     def _dual_new_ub(theta):
    #         lam1, lam2, lam3, lam4 = theta[:m], theta[m : 2 * m], theta[2 * m : 2 * m + n], theta[2 * m + n :]
    #         z = np.exp((E @ (lam2 - lam1) - lam3 + lam4) / kx_ub - 1)
    #         return z @ kx_ub - lam1 @ (1 - 1 / x_lb) - lam2 @ (1 / x_ub - 1) + lam3 @ z_ub - lam4 @ z_lb

    #     bounds = [(0, None)] * (2 * m + 2 * n)
    #     theta_start = np.random.uniform(size=2 * m + 2 * n)
    #     min_problem = minimize(
    #         _dual_new_ub, theta_start, bounds=bounds, method="SLSQP", options={"maxiter": 1e5, "ftol": 1e-12}
    #     )
    #     assert min_problem.success, min_problem
    #     # theta = min_problem.x
    #     # lam1, lam2, lam3, lam4 = theta[:m], theta[m : 2 * m], theta[2 * m : 2 * m + n], theta[2 * m + n :]
    #     # print("lam3: ", lam3)
    #     # print("lam4: ", lam4)
    #     return min_problem.fun

    # def new_upper_bound(self, cube):

    #     S = np.exp(self.problem.A)  # m * n

    #     z_lb = np.exp(-self.problem.p_ub * self.problem.b)
    #     z_ub = np.exp(-self.problem.p_lb * self.problem.b)

    #     z_start = np.asarray(self.lp["x"])[: self.n, 0] / SCALE
    #     z_start = np.maximum(z_start, z_lb)
    #     z_start = np.minimum(z_start, z_ub)

    #     x_ub = cube.center + cube.radius + 1e-12
    #     x_lb = cube.center - cube.radius - 1e-12

    #     def cnstr_1(z, c):
    #         return 1 / x_lb[c] - 1 - S[c] @ z  # >= 0

    #     def cnstr_2(z, c):
    #         return 1 + S[c] @ z - 1 / x_ub[c]  # >= 0

    #     constraints = [
    #         {"type": "ineq", "fun": lambda theta, c=c: cnstr_1(theta, c)} for c in range(self.m)
    #     ] + [{"type": "ineq", "fun": lambda theta, c=c: cnstr_2(theta, c)} for c in range(self.m)]

    #     bounds = list(zip(z_lb, z_ub))

    #     CONST = 1e6

    #     for c in range(self.m):
    #         for f in [cnstr_1, cnstr_2]:
    #             assert (
    #                 f(z_start, c) >= 0.0
    #             ), f"Starting value constraint violated, {f(z_start, c)} < 0 at {f}"

    #     for i in range(self.n):
    #         assert z_start[i] <= z_ub[i], f"{i} out of bounds (> z_ub)."
    #         assert z_start[i] >= z_lb[i], f"{i} out of bounds ({z_start[i]} < {z_lb[i]})."

    #     def objective(z):
    #         return np.sum(xlogy(z, z) * (self.problem.k @ x_ub)) / CONST

    #     def jac(z):
    #         z_ = np.maximum(z, z_lb)
    #         dfdz = (1 + np.log(z_)) * (self.problem.k @ x_ub)
    #         return dfdz / CONST

    #     opt = minimize(
    #         objective,
    #         z_start,
    #         bounds=bounds,
    #         jac=jac,
    #         constraints=constraints,
    #         options={"iprint": 99, "maxiter": 10000},
    #     )

    #     # assert opt.success, f"opt: {opt}, theta_start: {z_start}."
    #     if opt.success:
    #         return -opt.fun * CONST
    #     else:
    #         print("WARNING: new bound optimization failed.")
    #         return np.inf



        # def objective_fixed_x(z):
        #     return np.sum(xlogy(z, z) * (self.problem.k @ x_delta)) / 1e6

        # def jac_fixed_x(z):
        #     z_ = np.maximum(z, z_lb)
        #     dfdz = (1 + np.log(z_)) * (self.problem.k @ x_delta)
        #     return dfdz / 1e6

        # def cnstr_fixed_x(z, c):
        #     return 1 + S[c] @ z - 1 / x_delta[c]  # >= 0

        # for c in range(self.m):
        #     print(f"cnstr {c}: {cnstr_fixed_x(z_start_, c)}")

        # opt_fixed_x = minimize(
        #     objective_fixed_x,
        #     z_start_,
        #     bounds=list(zip(z_lb, z_ub)),
        #     jac=jac_fixed_x,
        #     constraints=[
        #         {"type": "eq", "fun": lambda z, c=c: cnstr_fixed_x(z, c)}
        #         for c in range(self.m)
        #     ]
        # )
        # assert opt_fixed_x.success, f"pi(x) failed: {opt_fixed_x}"
        # cube.lb_alt = opt_fixed_x.fun



          # def new_upper_bound(self, cube):

    #     z_lb = np.exp(-self.problem.p_ub * self.problem.b)
    #     print(z_lb)
    #     z_ub = np.exp(-self.problem.p_lb * self.problem.b)
    #     x_ub = cube.center + cube.radius
    #     x_lb = cube.center - cube.radius
    #     S = np.exp(self.problem.A)

    #     def cnstr_1(theta, c):
    #         z = theta[self.m:]
    #         return 1 / x_lb[c] - 1 - S[c] @ z  # >= 0

    #     def cnstr_2(theta, c):
    #         z = theta[self.m:]
    #         return 1 + S[c] @ z - 1 / x_ub[c]   # >= 0

    #     constraints = (
    #         [{"type": "ineq", "fun": lambda theta, c=c: cnstr_1(theta, c)} for c in range(self.m)] +
    #         [{"type": "ineq", "fun": lambda theta, c=c: cnstr_2(theta, c)} for c in range(self.m)]
    #     )

    #     theta_start = np.hstack(((x_ub + x_lb) / 2, (z_ub + z_lb) / 2))

    #     bounds = list(zip(x_lb, x_ub)) + list(zip(z_lb, z_ub))

    #     def objective(theta):
    #         x, z = theta[:self.m], theta[self.m:]
    #         try:
    #             return np.sum(z * np.log(z) * (self.problem.k @ x)) / 1e6
    #         except Warning:
    #             print(z)
    #             raise Exception

    #     opt = minimize(objective, theta_start, bounds=bounds, constraints=constraints)

    #     assert opt.success, opt
    #     return - opt.fun * 1e6

    # def compute_relaxation_upper_bound(self, cube):

    #     x_lb, x_ub = cube.center - cube.radius, cube.center + cube.radius
    #     n = self.n

    #     revenue_ub = []
    #     for c, segment in enumerate(self.problem.segments):

    #         x_lb_c, x_ub_c = x_lb[c], x_ub[c]

    #         def inequality_1(q):
    #             return np.sum(q) - 1 + x_ub_c

    #         def inequality_2(q):
    #             return 1 - np.sum(q) - x_lb_c

    #         constr = [
    #             {"type": "ineq", "fun": inequality_1},
    #             {"type": "ineq", "fun": inequality_2},
    #         ]
    #         with np.errstate(all='ignore'):
    #             opt = minimize(
    #                 lambda q, segment=segment: -segment.revenue(q),
    #                 np.zeros(n) + (x_ub_c + x_lb_c) / (2 * n),
    #                 bounds=[(0, 1)] * n,
    #                 constraints=constr
    #             )

    #         # print(1 - np.sum(opt.x))

    #         revenue_ub.append(- opt.fun)

    #     return np.sum(np.asarray(revenue_ub) * self.problem.w)



        # for c in range(self.m):
        #     for f in [cnstr_1, cnstr_2]:
        #         assert (
        #             f(theta_start, c) >= -1e-20
        #         ), f"Starting value constraint violated, {f(theta_start, c)} < 0"
        # for i in range(self.n):
        #     assert z_start[i] <= z_ub[i], f"{i} out of bounds (> z_ub)."
        #     assert z_start[i] >= z_lb[i], f"{i} out of bounds ({z_start[i]} < {z_lb[i]})."



        # new_ub = np.inf
        # print(f"ub: {lipschitz_upper_bound}, new_ub: {new_ub}.")

        # if np.min(cube.center - cube.radius) >= 0.01:
        # return lipschitz_upper_bound

        # relaxation_upper_bound = self.compute_relaxation_upper_bound(cube)
        # relaxation_upper_bound = np.inf

        # print(f"\n cube center: {cube.center} (radius: {cube.radius})")
        # if relaxation_upper_bound < lipschitz_upper_bound:
        #     print(f"relaxation better: {relaxation_upper_bound} vs {lipschitz_upper_bound}")
        # else:
        #     print(f"lipschitz better: {lipschitz_upper_bound} vs {relaxation_upper_bound}")


        # S = np.exp(self.problem.A)
        # print("z_unscaled: ", np.asarray(self.lp["x"])[: self.n, 0])
        # z = np.asarray(self.lp["x"])[: self.n, 0] / SCALE
        # x_delta, _ = self._get_x_r_delta(cube)
        # z_cnstr = np.maximum(z, self.z_lb)
        # print("1 / (1 + Sz) ", 1 / (1 + S @ z))
        # print("1 / (1 + Sz_cnstr) ", 1 / (1 + S @ z_cnstr))
        # print("x_delta ", x_delta)
