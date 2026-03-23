import pandas as pd

df = pd.read_csv("outputs/dae/per_task_metrics.csv")

# Filter test split only
test_df = df[df['split'] == 'test']

# Find participants with both structured and naturalistic tasks
structured_tasks  = ['cookie_theft', 'cinderella', 'sandwich']
naturalistic_tasks = ['conversation']

has_structured = set(
    test_df[test_df['task'].isin(structured_tasks)]['participant_id']
)
has_naturalistic = set(
    test_df[test_df['task'] == 'conversation']['participant_id']
)

matched = has_structured & has_naturalistic

print(f"Test participants with structured tasks:    {len(has_structured)}")
print(f"Test participants with conversation:        {len(has_naturalistic)}")
print(f"Test participants with BOTH:                {len(matched)}")