import itertools

# ==============================
# 1. Players and Strategy Definition
# ==============================
# Player U (Upstream) Strategy: 0=No-Plant, 1=Plant-Trees, 2=Invest-Levee
# Player D (Downstream) Strategy: 0=Do-Nothing, 1=Invest-Levee, 2=Relocation

strategies_U = ["No-Plant", "Plant-Trees", "Invest-Levee"]
strategies_D = ["Do-Nothing", "Invest-Levee", "Relocation"]

# ==============================
# 2. Payoff Matrix
# ==============================
# payoff_U[u][d] = Upstream payoff when Upstream strategy is u and Downstream strategy is d
# payoff_F[u][d] = Downstream payoff when Upstream strategy is u and Downstream strategy is d

payoff_U = [
    # U: 0(No-Plant) 
    [   0 ,  -2,   -3  ],  # D: 0,1,2
    # U: 1(Plant-Trees)
    [   2 ,   0,   -1  ],
    # U: 2(Invest-Levee) 
    [   0 ,   1,    3  ]
]

payoff_D = [
    # U: 0(No-Plant) 
    [   0 ,   3,    1  ],   # D: 0,1,2
    # U: 1(Plant-Trees) 
    [   1 ,   2,    2  ],
    # U: 2(Invest-Levee)
    [  -1 ,   1,    4  ]
]

# ==============================
# 3. Find Nash equilibrium
# ==============================
def find_nash_equilibria(payoff_U, payoff_D):
    n_u = len(payoff_U)      # Num of Strategy for U
    n_d = len(payoff_U[0])   # Num of Strategy for D

    nash_equilibria = []

    for u_strat in range(n_u):
        for d_strat in range(n_d):
            # check (u_strat, d_strat) is a nash equilibrium

            current_payoff_U = payoff_U[u_strat][d_strat]
            is_best_response_U = True
            for u_alt in range(n_u):
                if payoff_U[u_alt][d_strat] > current_payoff_U:
                    is_best_response_U = False
                    break

            current_payoff_D = payoff_D[u_strat][d_strat]
            is_best_response_D = True
            for f_alt in range(n_d):
                if payoff_D[u_strat][f_alt] > current_payoff_D:
                    is_best_response_D = False
                    break

            if is_best_response_U and is_best_response_D:
                nash_equilibria.append( (u_strat, d_strat) )

    return nash_equilibria


# ==============================
# 4. Solve Nash equilibrium
# ==============================

NE = find_nash_equilibria(payoff_U, payoff_D)

print("=== Nash Equilibrium Search ===")
for (u_idx, f_idx) in NE:
    print(f" - NE strategy pair: (U: {strategies_U[u_idx]}, D: {strategies_D[f_idx]})")
    print(f"   -> Payoff_U = {payoff_U[u_idx][f_idx]}, Payoff_D = {payoff_D[u_idx][f_idx]}")

if not NE:
    print("No pure-strategy Nash Equilibrium found.")