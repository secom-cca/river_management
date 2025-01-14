'''
Example: Upstream (U) vs. Downstream (D) with payoff depending on (environment, economy, safety) plus weights.
'''

# 1. Strategy labels
strategies_U = ["NP", "PT", "IL"]  # Upstream: No-Plant, Plant-Trees, Invest-Levee
strategies_D = ["DN", "IL", "RE"]  # Downstream: Do-Nothing, Invest-Levee, Relocation

# 2. Example (env, econ, safe) payoff dictionaries
#    for each (sU, sD) from the perspective of U and D.

# Upstream environment/economy/safety
env_U = {
    ("NP","DN"): -1, ("NP","IL"): -2, ("NP","RE"): -1,
    ("PT","DN"): +2, ("PT","IL"): +1, ("PT","RE"): +2,
    ("IL","DN"): -2, ("IL","IL"): -3, ("IL","RE"): -2
}
econ_U = {
    ("NP","DN"): +2, ("NP","IL"): +1, ("NP","RE"): +2,
    ("PT","DN"): -1, ("PT","IL"): -2, ("PT","RE"): -1,
    ("IL","DN"): -1, ("IL","IL"): -2, ("IL","RE"): -1
}
safe_U = {
    ("NP","DN"): 0,  ("NP","IL"): +1, ("NP","RE"): 0,
    ("PT","DN"): +1, ("PT","IL"): +2, ("PT","RE"): +1,
    ("IL","DN"): +2, ("IL","IL"): +3, ("IL","RE"): +2
}

# Downstream environment/economy/safety
env_D = {
    ("NP","DN"): 0,  ("NP","IL"): -1, ("NP","RE"): 0,
    ("PT","DN"): +1, ("PT","IL"): 0,  ("PT","RE"): +1,
    ("IL","DN"): -1, ("IL","IL"): -2, ("IL","RE"): -1
}
econ_D = {
    ("NP","DN"): 0,  ("NP","IL"): -1, ("NP","RE"): -2,
    ("PT","DN"): 0,  ("PT","IL"): -1, ("PT","RE"): -2,
    ("IL","DN"): -1, ("IL","IL"): -2, ("IL","RE"): -3
}
safe_D = {
    ("NP","DN"): -1, ("NP","IL"): +2, ("NP","RE"): +3,
    ("PT","DN"): 0,  ("PT","IL"): +2, ("PT","RE"): +3,
    ("IL","DN"): +1, ("IL","IL"): +3, ("IL","RE"): +4
}

# 3. Set weights for both players (example)
wU_env, wU_econ, wU_safe = 1, 1, 1   # Upstream weights
wD_env, wD_econ, wD_safe = 1, 1, 1   # Downstream weights

def payoff_U(sU, sD):
    return (wU_env * env_U[(sU,sD)]
          + wU_econ * econ_U[(sU,sD)]
          + wU_safe * safe_U[(sU,sD)])

def payoff_D(sU, sD):
    return (wD_env * env_D[(sU,sD)]
          + wD_econ * econ_D[(sU,sD)]
          + wD_safe * safe_D[(sU,sD)])

# 4. Find pure-strategy Nash equilibria
def find_nash_equilibria():
    nash_list = []
    for sU in strategies_U:
        for sD in strategies_D:
            # Check if sU is best response to sD
            u_val = payoff_U(sU, sD)
            u_best = True
            for altU in strategies_U:
                if payoff_U(altU, sD) > u_val:
                    u_best = False
                    break
            
            # Check if sD is best response to sU
            d_val = payoff_D(sU, sD)
            d_best = True
            for altD in strategies_D:
                if payoff_D(sU, altD) > d_val:
                    d_best = False
                    break
            
            if u_best and d_best:
                nash_list.append((sU, sD))
    return nash_list

if __name__ == "__main__":
    NE = find_nash_equilibria()
    if len(NE) == 0:
        print("No pure-strategy Nash Equilibrium found.")
    else:
        print("Pure-strategy Nash Equilibria:")
        for (sU, sD) in NE:
            print(f"  (U={sU}, D={sD}) -> (uU={payoff_U(sU,sD)}, uD={payoff_D(sU,sD)})")
