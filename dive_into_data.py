import pandas as pd
import pyarrow.parquet as pq
import numpy as np


f1 = "./openagentrl/Open-AgentRL-SFT-3K.code_only.parquet"
f2 = "./processed_trajectories/trajectories_messages.parquet"

# Read both files
print("=" * 80)
print("REFERENCE FILE (Open-AgentRL-SFT-3K.code_only.parquet)")
print("=" * 80)

df1 = pd.read_parquet(f1)
table1 = pq.read_table(f1)

print(f"\nShape: {df1.shape}")
print(f"\nColumns: {df1.columns.tolist()}")
print(f"\nData types:\n{df1.dtypes}")
print(f"\nPyArrow Schema:\n{table1.schema}")

print("\n" + "=" * 80)
print("YOUR FILE (filtered_code_act_instruct_sft.parquet)")
print("=" * 80)

df2 = pd.read_parquet(f2)
table2 = pq.read_table(f2)

print(f"\nShape: {df2.shape}")
print(f"\nColumns: {df2.columns.tolist()}")
print(f"\nData types:\n{df2.dtypes}")
print(f"\nPyArrow Schema:\n{table2.schema}")

print("\n" + "=" * 80)
print("SCHEMA COMPARISON")
print("=" * 80)

# Check column differences
cols1 = set(df1.columns)
cols2 = set(df2.columns)

print(f"\nColumns only in reference: {cols1 - cols2}")
print(f"Columns only in your file: {cols2 - cols1}")
print(f"Common columns: {cols1 & cols2}")

# Check data type differences for common columns
print("\n" + "=" * 80)
print("DATA TYPE MISMATCHES")
print("=" * 80)

for col in cols1 & cols2:
    type1 = table1.schema.field(col).type
    type2 = table2.schema.field(col).type
    
    if type1 != type2:
        print(f"\n❌ Column '{col}':")
        print(f"   Reference: {type1}")
        print(f"   Your file: {type2}")
    else:
        print(f"✓ Column '{col}': {type1}")

# Sample data inspection for 'messages' column
print("\n" + "=" * 80)
print("SAMPLE DATA - 'messages' column")
print("=" * 80)

if 'messages' in df1.columns:
    print("\nReference file - first 2 rows:")
    for i in range(min(2, len(df1))):
        msg = df1['messages'].iloc[i]
        msg_type = type(msg).__name__
        is_array = isinstance(msg, np.ndarray)
        is_list = isinstance(msg, list)
        length = len(msg) if hasattr(msg, '__len__') else 'N/A'
        print(f"  Row {i}: type={msg_type}, is_array={is_array}, is_list={is_list}, length={length}")
        if hasattr(msg, '__iter__') and not isinstance(msg, str):
            first_elem = next(iter(msg), None)
            if first_elem is not None:
                print(f"    First element type: {type(first_elem).__name__}")

if 'messages' in df2.columns:
    print("\nYour file - first 2 rows:")
    for i in range(min(2, len(df2))):
        msg = df2['messages'].iloc[i]
        msg_type = type(msg).__name__
        is_array = isinstance(msg, np.ndarray)
        is_list = isinstance(msg, list)
        length = len(msg) if hasattr(msg, '__len__') else 'N/A'
        print(f"  Row {i}: type={msg_type}, is_array={is_array}, is_list={is_list}, length={length}")
        if hasattr(msg, '__iter__') and not isinstance(msg, str):
            first_elem = next(iter(msg), None)
            if first_elem is not None:
                print(f"    First element type: {type(first_elem).__name__}")

# Check for null/mixed values in messages
print("\n" + "=" * 80)
print("NULL AND TYPE CONSISTENCY CHECK")
print("=" * 80)

if 'messages' in df2.columns:
    print("\nYour file 'messages' column:")
    print(f"  Null count: {df2['messages'].isna().sum()}")
    
    types_found = df2['messages'].apply(lambda x: type(x).__name__).value_counts()
    print(f"  Types found:\n{types_found}")
    
    # FIXED: Check for non-array and non-list values properly
    print("\n  Checking for unexpected types...")
    
    # Count numpy arrays
    num_arrays = df2['messages'].apply(lambda x: isinstance(x, np.ndarray)).sum()
    print(f"  Numpy arrays: {num_arrays}")
    
    # Count lists
    num_lists = df2['messages'].apply(lambda x: isinstance(x, list)).sum()
    print(f"  Python lists: {num_lists}")
    
    # Find anything that's neither array nor list
    def is_valid_type(x):
        return isinstance(x, (np.ndarray, list)) or pd.isna(x)
    
    invalid_mask = ~df2['messages'].apply(is_valid_type)
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        print(f"\n  ❌ WARNING: Found {invalid_count} rows with unexpected types")
        print(f"  Sample rows with unexpected types:")
        # Use .loc with boolean mask instead of using the Series as a key
        invalid_rows = df2.loc[invalid_mask]
        print(invalid_rows.head())
    else:
        print(f"\n  ✓ All rows have valid types (numpy.ndarray or list)")

# COMPARISON SUMMARY
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

if 'messages' in df1.columns and 'messages' in df2.columns:
    ref_types = df1['messages'].apply(lambda x: type(x).__name__).value_counts()
    your_types = df2['messages'].apply(lambda x: type(x).__name__).value_counts()
    
    print("\nReference file 'messages' types:")
    for type_name, count in ref_types.items():
        print(f"  {type_name}: {count}")
    
    print("\nYour file 'messages' types:")
    for type_name, count in your_types.items():
        print(f"  {type_name}: {count}")
    
    # Check if they match
    if ref_types.to_dict() == your_types.to_dict():
        print("\n✓ Type distributions MATCH - files are compatible!")
    else:
        print("\n⚠️  Type distributions DIFFER")
        print("    This may cause compatibility issues")