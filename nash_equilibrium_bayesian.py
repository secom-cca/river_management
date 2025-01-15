import itertools

# ================================================================
# 1. Define the sets of types and actions for each player
# ================================================================
types_U = ["U1", "U2"]  # Upstream has 2 possible types
types_D = ["D1", "D2"]  # Downstream has 2 possible types

actions_U = ["NP", "PT", "IL"]  # No-Plant, Plant-Trees, Invest-Levee
actions_D = ["DN", "IL", "RE"]  # Do-Nothing, Invest-Levee, Relocation

# ================================================================
# 2. Define the prior probability for each type
#    For simplicity, assume independence: p(U1)=p_U1, p(U2)=1-p_U1, etc.
# ================================================================
p_U1 = 0.5
p_U2 = 1.0 - p_U1
p_D1 = 0.4
p_D2 = 1.0 - p_D1

# A small helper function to get the joint probability:
def prob_joint(tU, tD):
    # If truly independent:
    if tU == "U1" and tD == "D1":
        return p_U1 * p_D1
    elif tU == "U1" and tD == "D2":
        return p_U1 * p_D2
    elif tU == "U2" and tD == "D1":
        return p_U2 * p_D1
    elif tU == "U2" and tD == "D2":
        return p_U2 * p_D2

# ================================================================
# 3. Define payoff dictionaries:
#    payoff_U[(sU, sD, tU, tD)] = Upstream's payoff
#    payoff_D[(sU, sD, tU, tD)] = Downstream's payoff
# ================================================================

payoff_U = {}
payoff_D = {}

# For each combination of (sU, sD, tU, tD), we define some payoff.
# Example approach: we create two sets of payoff tables, one for each type
# of Upstream, and combine them, etc.

# --- Example: We'll just define random or synthetic values:
import random
random.seed(42)  # for reproducibility

for sU in actions_U:
    for sD in actions_D:
        for tU in types_U:
            for tD in types_D:
                # We'll artificially define something like:
                # Base payoff from the pure (env, econ, safe) idea:
                # We'll skip the full environment/econ/safety table here
                # and just produce random-ish integers for demonstration.
                
                # E.g. type U1 might "like" environment more => payoff_U +2 if sU="PT"
                # This is purely illustrative.

                valU = 0
                valD = 0

                # Some naive logic: If tU=U1, bigger environment weight for "PT"
                if tU == "U1":
                    if sU == "PT": 
                        valU += 2
                    if sU == "IL": 
                        valU -= 1
                else:  # tU == "U2" => bigger economy weight for "NP"
                    if sU == "NP":
                        valU += 2
                    if sU == "IL":
                        valU += 1  # maybe type U2 sees some industrial benefit

                # If tD == "D1", might emphasize cost saving => prefer DN
                # If tD == "D2", might emphasize safety => prefer RE
                if tD == "D1":
                    if sD == "DN":
                        valD += 2
                    if sD == "RE":
                        valD -= 1
                else:  # tD == "D2"
                    if sD == "RE":
                        valD += 3
                    if sD == "IL":
                        valD += 1

                # Interactions: e.g. if (sU=IL, sD=IL), synergy or overlap
                # We'll add small random offset:
                valU += random.randint(-1, 1)
                valD += random.randint(-1, 1)

                payoff_U[(sU, sD, tU, tD)] = valU
                payoff_D[(sU, sD, tU, tD)] = valD


# ================================================================
# 4. Strategy: 
#    A pure strategy for Upstream is a function sigma_U: T_U -> S_U
#    Similarly for Downstream. 
#    We'll enumerate all possible such functions for each side,
#    then check if they're mutual best responses in the Bayesian sense.
# ================================================================

def all_pure_strategies(actions, types):
    """
    Return a list of all functions from 'types' to 'actions'.
    Represented as a dict: {t: a, ...} or a tuple in some canonical order.
    """
    # For 2 types & 3 actions, we have 3^2=9 possible strategy functions.
    # We'll yield them as dictionaries: {type -> action}
    all_strats = []
    for mapping in itertools.product(actions, repeat=len(types)):
        # mapping is a tuple like ('NP','IL') meaning T1->NP, T2->IL
        strategy_dict = {}
        for i, t in enumerate(types):
            strategy_dict[t] = mapping[i]
        all_strats.append(strategy_dict)
    return all_strats

sigmaU_candidates = all_pure_strategies(actions_U, types_U)
sigmaD_candidates = all_pure_strategies(actions_D, types_D)

def expected_payoff_U(sigmaU, sigmaD, tU):
    """
    Upstream's expected payoff given that Upstream is type tU,
    and Downstream's strategy function is sigmaD, 
    summing over Downstream's possible types with probabilities.
    """
    total = 0.0
    for tD in types_D:
        p = prob_joint(tU, tD)  # up to the factor p_U(tU) * p_D(tD)
        sU = sigmaU[tU]
        sD = sigmaD[tD]
        total += payoff_U[(sU, sD, tU, tD)] * (p / p_U1 if tU=="U1" else p / p_U2)
        # Explanation: we want E_{tD}[ payoff_U(...) ]
        # If tU is fixed, the prob distribution over tD is p(tD)/sum_{tD} p(tD),
        # but sum_{tD} p(tU, tD) = p(tU), so we divide by p(tU).
    return total

def expected_payoff_D(sigmaU, sigmaD, tD):
    """
    Downstream's expected payoff given that Downstream is type tD,
    and Upstream's strategy function is sigmaU,
    summing over Upstream's possible types with probabilities.
    """
    total = 0.0
    for tU in types_U:
        p = prob_joint(tU, tD)
        sU = sigmaU[tU]
        sD = sigmaD[tD]
        if tD=="D1":
            total += payoff_D[(sU, sD, tU, tD)] * (p / (p_D1))
        else:
            total += payoff_D[(sU, sD, tU, tD)] * (p / (p_D2))
    return total

def is_best_response_U(sigmaU, sigmaD):
    """
    For each type tU, check if sigmaU(tU) is a best response 
    among all possible actions in actions_U.
    That is, we fix the other player's strategy function sigmaD,
    and each type tU tries all possible aU in actions_U to see
    if sigmaU(tU) yields the highest expected payoff.
    """
    for tU in types_U:
        current_action = sigmaU[tU]
        current_val = expected_payoff_U(sigmaU, sigmaD, tU)
        # Check if any other action yields a strictly higher payoff:
        for aU in actions_U:
            if aU == current_action:
                continue
            # Consider a hypothetical strategy func where we replace
            # sigmaU(tU) with aU, but keep others the same.
            alt_sigmaU = dict(sigmaU)
            alt_sigmaU[tU] = aU
            alt_val = expected_payoff_U(alt_sigmaU, sigmaD, tU)
            if alt_val > current_val:
                return False  # Not best response for type tU
    return True

def is_best_response_D(sigmaU, sigmaD):
    """
    Same logic for Downstream.
    """
    for tD in types_D:
        current_action = sigmaD[tD]
        current_val = expected_payoff_D(sigmaU, sigmaD, tD)
        for aD in actions_D:
            if aD == current_action:
                continue
            alt_sigmaD = dict(sigmaD)
            alt_sigmaD[tD] = aD
            alt_val = expected_payoff_D(sigmaU, alt_sigmaD, tD)
            if alt_val > current_val:
                return False
    return True

def find_bayes_nash_equilibria():
    """
    Enumerate all pure strategy profiles (sigmaU, sigmaD) 
    and check if they form a Bayes--Nash equilibrium.
    """
    bne_list = []
    for sU_func in sigmaU_candidates:
        for sD_func in sigmaD_candidates:
            if is_best_response_U(sU_func, sD_func) and is_best_response_D(sU_func, sD_func):
                bne_list.append((sU_func, sD_func))
    return bne_list

if __name__ == "__main__":
    bne = find_bayes_nash_equilibria()
    if not bne:
        print("No pure-strategy Bayes-Nash Equilibrium found.")
    else:
        print(f"Found {len(bne)} pure-strategy Bayes-Nash Equilibria:")
        for i, (su, sd) in enumerate(bne, 1):
            print(f" BNE #{i}")
            print("  U's strategy function:", su)
            print("  D's strategy function:", sd)
