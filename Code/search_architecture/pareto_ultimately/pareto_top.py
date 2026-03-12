import pandas as pd
import numpy as np

from Code.search_architecture.darts_search.run_darts_search import TOP_K


db_name = 'e'

csv_file = f'Code/search_architecture/results/{db_name}_results.csv'
df = pd.read_csv(csv_file)

print(f"📊 Loaded {len(df)} records")


def pareto_front(df, reward_col='combined_reward', param_col='parameters_M'):
    df_temp = df.copy()
    df_temp['neg_param'] = -df_temp[param_col]

    points = df_temp[[reward_col, 'neg_param']].values
    n = len(points)

    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                if (points[j][0] >= points[i][0]) and (points[j][1] >= points[i][1]):
                    is_pareto[i] = False
                    break

    return df[is_pareto].copy()

pareto_df = pareto_front(df)
print(f"🎯 Found {len(pareto_df)} Pareto optimal solutions")


weight_reward = 0.5
weight_param = 0.5

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized = scaler.fit_transform(pareto_df[['combined_reward', 'parameters_M']])

normalized[:, 1] = 1 - normalized[:, 1]

score = weight_reward * normalized[:, 0] + weight_param * normalized[:, 1]
pareto_df['score'] = score

top_k = 20

top = pareto_df.sort_values('score', ascending=False).head(top_k)

print(f"Pareto front calculated with {len(top)} solutions.")
print(f"Although weighted comprehensive score selects {top_k} top, Pareto front size may be smaller.")
print("\n🏆 Top 5 solutions by weighted score within Pareto front:")

print("=" * 80)
for idx, row in top.iterrows():
    print(f"Rank: {idx + 1}")
    print(f"  combined_reward: {row['combined_reward']:.6f}")
    print(f"  parameters_M: {row['parameters_M']:.6f}")
    print(f"  architecture: {row['architecture']}")
    print(f"  source_file: {row['source_file']}")
    print("-" * 80)