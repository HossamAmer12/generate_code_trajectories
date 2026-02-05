import pandas as pd

f1    =  "./openagentrl/Open-AgentRL-SFT-3K.code_only.parquet"
f2    = "./processed_trajectories/trajectories_messages.parquet"
oname = "./processed_trajectories/merged_trajectories_messages_sft.parquet"

df1 = pd.read_parquet(f1)
df2 = pd.read_parquet(f2)

merged = pd.concat([df1, df2], ignore_index=True)

merged.to_parquet(oname, engine="pyarrow")

print(oname)