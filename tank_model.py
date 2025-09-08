from scipy.optimize import minimize

def two_layer_tank_model(P, a1, a2, c):
    S1, S2 = 0.0, 0.0
    q1_list, q2_list, level_list = [], [], []

    for p in P:
        q1 = a1 * S1
        i = c * S1
        S1 = S1 + p - q1 - i
        S1 = max(S1, 0)

        q2 = a2 * S2
        S2 = S2 + i - q2
        S2 = max(S2, 0)

        q1_list.append(q1)
        q2_list.append(q2)
        level_list.append(S2)

    q1_array = np.array(q1_list)
    q2_array = np.array(q2_list)
    level_array = np.array(level_list)
    flow_array = q1_array + q2_array
    return flow_array, level_array

# 損失関数（MSE）
def loss(params, P, true_flow, true_level):
    a1, a2, c = params
    flow_pred, level_pred = two_layer_tank_model(P, a1, a2, c)
    flow_mse = np.mean((true_flow - flow_pred) ** 2)
    level_mse = np.mean((true_level - level_pred) ** 2)
    return flow_mse + level_mse  # 合成誤差

# 初期値 & 最適化
initial_params = [0.1, 0.01, 0.05]
bounds = [(0, 1), (0, 1), (0, 1)]

P = merged_df['precipitation'].values
true_flow = merged_df['flow'].values
true_level = merged_df['level'].values

result = minimize(loss, initial_params, args=(P, true_flow, true_level), bounds=bounds)

opt_params = result.x
print("最適化されたパラメータ:", opt_params)

# 最終予測
flow_pred, level_pred = two_layer_tank_model(P, *opt_params)

# 可視化
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(true_flow, label='True Flow')
plt.plot(flow_pred, label='Tank Model Flow')
plt.title('Flow Comparison')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(true_level, label='True Level')
plt.plot(level_pred, label='Tank Model Level')
plt.title('Level Comparison')
plt.legend()

plt.tight_layout()
plt.show()
