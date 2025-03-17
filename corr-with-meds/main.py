import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Constants for productivity file
START_TIME_FIELD = " Start time"
END_TIME_FIELD = " End time"
TASK_NAME_FIELD = "Task name"
DURATION_FIELD = " Duration"
DATE_FIELD = "Date"
MEDS_TAKEN_FIELD = "Meds Taken"


def duration_to_hours(x: str):
    h, m, s = map(int, x.split(":"))
    return h + m / 60 + s / 3600


# Load the data
meds_df = pd.read_csv("meds.csv")
productivity_df = pd.read_csv("productivity.csv")

# Get dates for productivity_df
prod_first_date = datetime.strptime(
    productivity_df[START_TIME_FIELD].iloc[0], "%b %d, %Y at %I:%M:%S %p"
)
prod_last_date = datetime.strptime(
    productivity_df[START_TIME_FIELD].iloc[-1], "%b %d, %Y at %I:%M:%S %p"
)

meds_first_date = datetime.strptime(meds_df[DATE_FIELD].iloc[0], "%Y-%m-%d")
meds_last_date = datetime.strptime(meds_df[DATE_FIELD].iloc[-1], "%Y-%m-%d")

# Get common date range
first_date = max(prod_first_date, meds_first_date)
last_date = min(prod_last_date, meds_last_date)

first_date_str = first_date.strftime("%Y-%m-%d")
last_date_str = last_date.strftime("%Y-%m-%d")

# Preprocess productivity_df
productivity_df.drop([" Task description", " Note", " Tag"], axis=1, inplace=True)
productivity_df[END_TIME_FIELD] = productivity_df[END_TIME_FIELD].apply(
    lambda x: datetime.date(
        datetime.strptime(x, "%b %d, %Y at %I:%M:%S %p")
    ).isoformat()
)
productivity_df[DURATION_FIELD] = productivity_df[DURATION_FIELD].apply(
    duration_to_hours
)
productivity_df = productivity_df[
    (productivity_df[END_TIME_FIELD] >= first_date_str)
    & (productivity_df[END_TIME_FIELD] <= last_date_str)
]
productivity_df[TASK_NAME_FIELD] = productivity_df[TASK_NAME_FIELD].apply(
    lambda x: (
        "Productive" if x in ["Working", "Programming", "Studying", "Research"] else x
    )
)

daily_productive = (
    productivity_df[productivity_df[TASK_NAME_FIELD] == "Productive"]
    .groupby(END_TIME_FIELD)[DURATION_FIELD]
    .sum()
    .reset_index()
)
daily_productive = daily_productive.sort_values(END_TIME_FIELD)
daily_productive.columns = [DATE_FIELD, DURATION_FIELD]
productivity_df = daily_productive

# Add MedsTaken field
meds_df[MEDS_TAKEN_FIELD] = meds_df[meds_df.columns[-1]].apply(
    lambda x: int(x in ["Y", "y"])
)

# Ensure dates are in datetime format
meds_df[DATE_FIELD] = pd.to_datetime(meds_df[DATE_FIELD])
productivity_df[DATE_FIELD] = pd.to_datetime(productivity_df[DATE_FIELD])

# Merge the dataframes on Date
merged_df = pd.merge(meds_df, productivity_df, on=DATE_FIELD, how="inner")

# Display the first few rows to verify the merge
print("Merged DataFrame:")
print(merged_df.head())
print("=" * 50)
print()

## Section 1. Correlation analysis
# Calculate correlation between MedsTaken and ProductiveTime
correlation, p_value = stats.pointbiserialr(
    merged_df[MEDS_TAKEN_FIELD], merged_df[DURATION_FIELD]
)

print(f"Point-biserial correlation coefficient: {correlation:.2f}")
print(f"P-value: {p_value:.3f}")

# Visualize the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x=MEDS_TAKEN_FIELD, y=DURATION_FIELD, data=merged_df)
plt.title("Medication vs Productivity")
plt.xlabel("Medication Taken")
plt.ylabel("Productive Time")
plt.grid(True)
plt.show()

# Group by medication status
meds_taken = merged_df[merged_df[MEDS_TAKEN_FIELD] == 1][DURATION_FIELD]
meds_not_taken = merged_df[merged_df[MEDS_TAKEN_FIELD] == 0][DURATION_FIELD]

# Perform t-test
t_stat, p_value_ttest = stats.ttest_ind(meds_taken, meds_not_taken)

print(f"T-test statistic: {t_stat:.4f}")
print(f"P-value: {p_value_ttest:.4f}")

# Visualize with boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=MEDS_TAKEN_FIELD, y=DURATION_FIELD, data=merged_df)
plt.title("Productivity by Medication Status")
plt.xlabel("Medication Taken (0=No, 1=Yes)")
plt.ylabel("Productive Time")
plt.grid(True)
plt.show()

## Section 2. Regression analysis
# Create a simple linear regression model
model = sm.OLS(
    merged_df[DURATION_FIELD], sm.add_constant(merged_df[MEDS_TAKEN_FIELD])
).fit()
print(model.summary())

# If there are other variables that might affect productivity, we should control for them
# For example, if we had a 'DayOfWeek' variable:
# merged_df['DayOfWeek'] = merged_df['Date'].dt.dayofweek
# model_with_controls = sm.OLS(merged_df['ProductiveTime'],
#                             sm.add_constant(pd.get_dummies(merged_df[['MedsTaken', 'DayOfWeek']],
#                                                          drop_first=True))).fit()
# print(model_with_controls.summary())

## Section 3. Causation analysis
# We can attempt Granger causality test, but with caveats
# Sort data by date to ensure time-series order
merged_df = merged_df.sort_values(DATE_FIELD)

# Prepare data for Granger test
data = merged_df[[MEDS_TAKEN_FIELD, DURATION_FIELD]].values

# Run Granger causality test with different lags
max_lag = 3
test_result = grangercausalitytests(data, max_lag, verbose=False)

# Extract and display p-values for each lag
for i in range(1, max_lag + 1):
    p_value = test_result[i][0]["ssr_ftest"][1]
    print(f"Granger causality p-value (lag={i}): {p_value:.4f}")

print(
    "\nNote on causation: Granger causality tests temporal precedence, not true causation."
)

## Section 4. Analyze lag effect of medication
# Create lag features
merged_df = merged_df.sort_values(DATE_FIELD)
merged_df[f"{MEDS_TAKEN_FIELD}_Lag1"] = merged_df[MEDS_TAKEN_FIELD].shift(1)
merged_df[f"{MEDS_TAKEN_FIELD}_Lag2"] = merged_df[MEDS_TAKEN_FIELD].shift(2)

print("=" * 50)
print(merged_df.head())
print("=" * 50)

# Drop rows with NaN values (first two days)
lagged_df = merged_df.dropna()

# Cast to int
lagged_df[f"{MEDS_TAKEN_FIELD}_Lag1"] = lagged_df[f"{MEDS_TAKEN_FIELD}_Lag1"].astype(
    int
)
lagged_df[f"{MEDS_TAKEN_FIELD}_Lag2"] = lagged_df[f"{MEDS_TAKEN_FIELD}_Lag2"].astype(
    int
)

# Analyze the relationship between MedsTaken_Lag1 and ProductiveTime
corr_lag1, p_value_lag1 = stats.pearsonr(
    lagged_df[f"{MEDS_TAKEN_FIELD}_Lag1"], lagged_df[DURATION_FIELD]
)
print(f"Correlation between MedsTaken(i-1) and ProductiveTime(i): {corr_lag1:.4f}")
print(f"P-value: {p_value_lag1:.4f}")

# Analyze the relationship between MedsTaken_Lag2 and ProductiveTime
corr_lag2, p_value_lag2 = stats.pearsonr(
    lagged_df[f"{MEDS_TAKEN_FIELD}_Lag2"], lagged_df[DURATION_FIELD]
)
print(f"Correlation between MedsTaken(i-2) and ProductiveTime(i): {corr_lag2:.4f}")
print(f"P-value: {p_value_lag2:.4f}")

# Multiple regression with both lags
model_lags = sm.OLS(
    lagged_df[DURATION_FIELD],
    sm.add_constant(
        lagged_df[
            [MEDS_TAKEN_FIELD, f"{MEDS_TAKEN_FIELD}_Lag1", f"{MEDS_TAKEN_FIELD}_Lag2"]
        ]
    ),
).fit()
print(model_lags.summary())

# If MedsTaken is binary, we can also analyze the average productivity for different patterns
if set(merged_df[MEDS_TAKEN_FIELD].unique()) == {0, 1}:
    # Create a pattern column (e.g., "101" means medication was taken on i and i+2 but not i+1)
    lagged_df["Pattern"] = (
        lagged_df[MEDS_TAKEN_FIELD].astype(str)
        + lagged_df[f"{MEDS_TAKEN_FIELD}_Lag1"].astype(str)
        + lagged_df[f"{MEDS_TAKEN_FIELD}_Lag2"].astype(str)
    )
    lagged_df_cols = list(lagged_df.columns)
    lagged_df_cols[lagged_df_cols.index(DURATION_FIELD)] = DURATION_FIELD.strip()
    lagged_df.columns = lagged_df_cols

    # Calculate mean productivity for each pattern
    pattern_means = lagged_df.groupby("Pattern")[DURATION_FIELD.strip()].agg(
        ["mean", "count"]
    )
    print("\nMean productivity by medication pattern (current, i-1, i-2):")
    print(pattern_means)

    # ANOVA to test if the patterns have significantly different means
    formula = f"{DURATION_FIELD} ~ C(Pattern)"
    model = ols(formula, data=lagged_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nANOVA results for medication patterns:")
    print(anova_table)

    # Post-hoc test if ANOVA is significant
    if anova_table.loc["C(Pattern)", "PR(>F)"] < 0.05:
        posthoc = pairwise_tukeyhsd(lagged_df, DURATION_FIELD, "Pattern")
        print("\nTukey's HSD post-hoc test results:")
        print(posthoc)
