# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import PercentFormatter

df = pd.read_csv("credit_risk_dataset.csv", sep = ";")
df
# %%

# Quantitative Analysis of Variables-------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------

df.describe()

# Data Cleaning and Feature Engineering----------------------------------------------------------------------------------------------------------
# %%
#Eliminating the Outliers of Age Person and person emp length
df = df[
    (df["person_age"] <= 100)
    & ((df["person_emp_length"] <= 60) | df["person_emp_length"].isna())
]
df.describe()
df

# %%
# 1) Flaging which rows were missing
df["int_rate_missing"] = df["loan_int_rate"].isna().astype(int)
df["emp_length_missing"] = df["person_emp_length"].isna().astype(int)

# 2) Impute with the mean of the same grade / overall median
df["loan_int_rate"] = df["loan_int_rate"].fillna(
    df.groupby("loan_grade", observed=True)["loan_int_rate"].transform("mean")
)
df["person_emp_length"] = df["person_emp_length"].fillna(
    df["person_emp_length"].median()
)

df["loan_int_rate"].isna().sum()
df["person_emp_length"].isna().sum()
df

round(df.describe(),2)

# Categorical Variable Encoding------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
# %%
Loan_Grade_Order = sorted(df["loan_grade"].dropna().unique())

# 1) Binary
df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"N": 0, "Y": 1})

# 2) Ordinal
grade_map = {g: i for i, g in enumerate(Loan_Grade_Order)}
df["loan_grade_num"] = df["loan_grade"].astype(str).map(grade_map)

# 3) One-hot
intent_dummies = pd.get_dummies(df["loan_intent"], prefix="loan_intent",
                                drop_first=True, dtype=int)
home_dummies = pd.get_dummies(df["person_home_ownership"], prefix="person_home_ownership",
                              drop_first=True, dtype=int)
df = pd.concat([df, intent_dummies, home_dummies], axis=1)
df

#Creating New Variables-----------------------------------------------------------------------------------------------------------------------------------
# %%
#Loan Detail Ratios
df["Annual_Interest"] = (df["loan_amnt"]*df["loan_int_rate"])/100
df["Interest_to_Income"] = df["Annual_Interest"] / df["person_income"]

# Age Ratios
df["cred_hist_to_age"] = df["cb_person_cred_hist_length"] / df["person_age"]

# Interactions
df["prior_default_low_grade"] = (df["cb_person_default_on_file"]
                                 * (df["loan_grade_num"] >= 3).astype(int))
# %%
# Visualization of variables---------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------

grade_palette = [
    "#3B0A0A",  # Very dark burgundy
    "#6E1414",  # Burgundy
    "#A11D1D",  # Deep red
    "#CF3A3A",  # Red
    "#E8716F",  # Coral
    "#F4A6A3",  # Salmon pink
    "#FBD7D5",  # Pale pink
]
grade_markers = ["o", "X", "s", "D", "v", "^", "P"]

intent_palette = [
    "#E4003A",  # Red
    "#00767A",  # Dark teal
    "#FF9EBB",  # Pink
    "#1A1A1A",  # Charcoal
    "#7F0020",  # Maroon
    "#8FD1D4",  # Light cyan
    "#A0A0A0",  # Grey
]

intent_markers = ["o", "s", "^", "D", "v", "X"]
# %%
grades_abc = ["A", "B", "C"]
df_abc = df[df["loan_grade"].isin(grades_abc)]
df_rest = df[~df["loan_grade"].isin(grades_abc)]

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

g = sns.catplot(data=df_abc, x="loan_grade", y="person_income", kind="violin",
                order=sorted(df_abc["loan_grade"].unique()))
ax = g.ax
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax.ticklabel_format(style="plain", axis="y")

ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000):,}k"))

plt.suptitle("Person Income by Loan Grade (A, B, C)", y=1.02)
plt.show()

sns.catplot(data=df_rest, x="loan_grade", y="person_income", kind="box", order=sorted(df_rest["loan_grade"].unique()))
plt.suptitle("Person Income by Loan Grade (D and beyond)", y=1.02)
plt.show()

# %%
sns.catplot(data=df, x="loan_grade", y="loan_percent_income", kind="box", order=Loan_Grade_Order)
plt.show()

# %%
intent_map = {
    "PERSONAL": "Personal",
    "EDUCATION": "Educ.",
    "MEDICAL": "Medical",
    "VENTURE": "Venture",
    "HOMEIMPROVEMENT": "Home Impr.",
    "DEBTCONSOLIDATION": "Debt Cons.",
}

df["loan_intent"] = df["loan_intent"].map(intent_map)
# %%
order = df["loan_intent"].value_counts().index

fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(data=df, x="loan_intent", order=order, color="#4C71B0", ax=ax)

# Put the count on top of each bar
ax.bar_label(ax.containers[0], fmt="{:,.0f}", padding=3, fontsize=10, color="#333333")

# Titles
ax.set_title("Loan Applications by Intent", fontsize=14, weight="bold", loc="left", pad=15)
ax.set_xlabel("Loan Intent", fontsize=11, color="#333333", labelpad=10)
ax.set_ylabel("")

# Remove the y-axis and the chart border
ax.get_yaxis().set_visible(False)
sns.despine(left=True)

# Clean up the x-axis labels
ax.set_xticks(range(len(order)))
ax.set_xticklabels([label.replace("_", " ").title() for label in order], fontsize=10)
ax.tick_params(axis="x", length=0)

plt.tight_layout()
plt.show()

# %%
g = sns.relplot(
    data=df,
    x="person_age",
    y="loan_percent_income",
    hue="loan_grade",
    hue_order=Loan_Grade_Order,
    style="loan_grade",
    style_order=Loan_Grade_Order,
    palette=intent_palette,
    markers=grade_markers,
    alpha=0.85,
    s=80,
    height=6,
    aspect=1.3
)

# Axis labels and title
g.set(
    xlabel="Person Age",
    ylabel="Loan Income Percentage",
    title="Loan Income Percentage by Age and Loan Grade"
)

# Clean up the legend
g._legend.set_title("Loan Grade")
g._legend.set_bbox_to_anchor((1.02, 1))
g._legend.set_loc("upper left")

# Grid
g.ax.grid(True, alpha=0.2)
g.ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

# %%
g = sns.relplot(
    data=df,
    x="person_age",
    y="person_emp_length",
    hue="loan_grade",
    hue_order=Loan_Grade_Order,
    style="loan_grade",
    style_order=Loan_Grade_Order,
    palette=intent_palette,
    markers=grade_markers,
    alpha=0.85,
    s=80,
    height=6,
    aspect=1.3
)

# Axis labels and title
g.set(
    xlabel="Person Age",
    ylabel="Employment Length (Years)",
    title="Employment Length by Age and Loan Grade"
)

# Clean up the legend
g._legend.set_title("Loan Grade")
g._legend.set_bbox_to_anchor((1.02, 1))
g._legend.set_loc("upper left")

# Add subtle grid
g.ax.grid(True, alpha=0.2)
g.ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

# %%
g = sns.relplot(
    data=df,
    x="person_age",
    y="cb_person_cred_hist_length",
    hue="loan_grade",
    hue_order=Loan_Grade_Order,
    style="loan_grade",
    style_order=Loan_Grade_Order,
    palette=grade_palette,
    markers=grade_markers,
    alpha=0.85,
    s=80,
    height=6,
    aspect=1.3
)

# Axis labels and title
g.set(
    xlabel="Person Age",
    ylabel="Credit History Length (Years)",
    title="Credit History Length by Age and Loan Grade"
)

# Clean up the legend
g._legend.set_title("Loan Grade")
g._legend.set_bbox_to_anchor((1.02, 1))
g._legend.set_loc("upper left")

# Add subtle grid
g.ax.grid(True, alpha=0.2)
g.ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
# %%
num_cols = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "cb_person_default_on_file", "loan_grade_num", "loan_status",
    "Annual_Interest", "Interest_to_Income", "cred_hist_to_age",
    "prior_default_low_grade"
] + list(intent_dummies.columns) + list(home_dummies.columns)

# Calculate correlation matrix
corr = df[num_cols].corr()

# Create readable labels for the variables
variable_labels = {
    "person_age": "Age",
    "person_income": "Annual Income",
    "person_emp_length": "Employment Length",
    "loan_amnt": "Loan Amount",
    "loan_int_rate": "Loan Interest Rate",
    "loan_percent_income": "Loan-to-Income Ratio",
    "cb_person_cred_hist_length": "Credit History Length",
    "cb_person_default_on_file": "Previous Default",
    "loan_grade_num": "Loan Grade",
    "loan_status": "Loan Status",
    "Annual_Interest": "Annual Interest",
    "Interest_to_Income": "Interest-to-Income Ratio",
    "cred_hist_to_age": "Credit History-to-Age Ratio",
    "prior_default_low_grade": "Previous Default & Low Grade"
}

# Add readable names for dummy variables
for col in intent_dummies.columns:
    variable_labels[col] = col.replace("_", " ").title()

for col in home_dummies.columns:
    variable_labels[col] = col.replace("_", " ").title()

# Rename rows and columns for the heatmap only
corr_display = corr.rename(
    index=variable_labels,
    columns=variable_labels
)

# Mask upper triangle and diagonal
mask = np.triu(np.ones_like(corr_display, dtype=bool))

# Create figure
plt.figure(figsize=(16, 13))

# Heatmap
sns.heatmap(
    corr_display,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={
        "shrink": 0.8,
        "label": "Correlation"
    },
    annot_kws={
        "size": 7
    }
)

# Title
plt.title(
    "Correlation Matrix of Loan and Applicant Features (incl. Dummy Variables)",
    fontsize=16,
    fontweight="bold",
    pad=20
)

# Axis labels
plt.xlabel("Variables", fontsize=11, labelpad=10)
plt.ylabel("Variables", fontsize=11, labelpad=10)

# Tick labels
plt.xticks(
    rotation=45,
    ha="right",
    fontsize=9
)

plt.yticks(
    rotation=0,
    fontsize=9
)

plt.tight_layout()
plt.show()
# %% Histogram
g = sns.displot(
    data=df, x="loan_percent_income", hue="loan_grade", hue_order=Loan_Grade_Order,
    palette=intent_palette, kind="hist", bins=20, multiple="stack",
    edgecolor="white", linewidth=0.5, height=5, aspect=1.8,
)

# Set X label and clear Y label
g.set_axis_labels("Loan Amount as % of Income", "", fontsize=11, color="#333333")
g.ax.xaxis.set_major_formatter(PercentFormatter(1.0))
g.ax.tick_params(length=0, colors="#555555")

# Completely hide the Y-axis (ticks and labels)
g.ax.yaxis.set_visible(False)
sns.despine(left=True)

# Calculate total stacked height per bin and display count on top of each bar
totals = {}
for p in g.ax.patches:
    x_center = p.get_x() + p.get_width() / 2
    top = p.get_y() + p.get_height()
    key = round(x_center, 6)
    totals[key] = max(totals.get(key, 0), top)

for x, total in totals.items():
    if total > 0:
        g.ax.annotate(
            f"{int(total)}",
            xy=(x, total),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
            weight="bold",
        )

# Expand top margin slightly so numbers don't touch the top edge
g.ax.set_ylim(top=g.ax.get_ylim()[1] * 1.06)

# Legend and centered title
g.legend.set_title("Loan Grade")
g.figure.suptitle(
    "Loan-to-Income Ratio by Loan Grade",
    fontsize=14, weight="bold", x=0.5, ha="center", y=1.03
)

plt.show()
# %%
g = sns.displot(
    data=df, x="loan_percent_income", hue="loan_grade", hue_order=Loan_Grade_Order,
    palette=intent_palette, kind="kde", fill=True, common_norm=False,
    alpha=0.25, linewidth=1.5, height=5, aspect=1.8,
)
g.set_axis_labels("Loan Amount as % of Income", "", fontsize=11, color="#333333")
g.ax.xaxis.set_major_formatter(PercentFormatter(1.0))
g.ax.get_yaxis()
g.ax.tick_params(length=0, colors="#555555")
sns.despine(left=False)
g.legend.set_title("Loan Grade")
g.figure.suptitle("Distribution of Loan-to-Income Ratio by Loan Grade",
                  fontsize=14, weight="bold", x=0.02, ha="left", y=1.03)
plt.show()

# %% 
# Loan Status counts by Loan Intent
# Orden por conteo total (status 0 + status 1), de mayor a menor
intent_order = (df["loan_intent"].value_counts().index.tolist())

ax = sns.countplot(data=df, x="loan_status", hue="loan_intent",
                    hue_order=intent_order, palette=intent_palette,
                    edgecolor="white", linewidth=0.5)

for container in ax.containers:
    ax.bar_label(container, fontsize=8, padding=2)

ax.set_title("Loan Status Counts by Loan Intent")
ax.legend(title="Loan Intent")

# Quita el título del eje x
ax.set_xlabel("")

# Cambia los ticks 0/1 por texto
ax.set_xticks([0, 1])
ax.set_xticklabels(["No Default", "Default"])

# Quita el eje y por completo
ax.set_ylabel("")
ax.get_yaxis().set_visible(False)
ax.spines["left"].set_visible(False)
ax.margins(y=0.1)

plt.show()
#%%
ax = sns.histplot(data=df_abc, x="person_income", hue="loan_grade",
                  hue_order=grades_abc, stat="density",
                  element="step", fill=True, common_norm=False, alpha=0.4,
                  log_scale=True)

ax.set_xlabel("Person Income (log scale)")
ax.set_ylabel("Density")
ax.set_title("Person Income Distribution by Loan Grade (A, B, C)")
plt.show()

#Relationship between Variables---------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------


# %%

# Mapping to make the text of the graph titles fit properly
home_map = {
    "RENT": "Rent",
    "OWN": "Own",
    "MORTGAGE": "Mortgage",
    "OTHER": "Other",
}

plot_df = df.assign(
    loan_intent=df["loan_intent"].replace(intent_map),
    person_home_ownership=df["person_home_ownership"].replace(home_map),
)

g = sns.FacetGrid(
    plot_df,
    row="person_home_ownership",
    col="loan_intent",
    hue = "person_home_ownership",
    palette="Set2",
    margin_titles=True,   # nombres de las filas a la derecha
    height=2.5,
    aspect=1.1,
)
g.map(sns.regplot, "loan_grade_num", "loan_int_rate",
      scatter_kws={"s": 10, "alpha": 0.4},
      line_kws= {"color": "lightblue", "linewidth": 2})

g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels("Loan grade (A=0 … G=6)", "Interest rate (%)")

plt.show()
# %%
sns.regplot(data=df, x="person_age", y="cred_hist_to_age",
            scatter_kws={"color": "green"},
            line_kws={"color": "red"})
plt.xlabel("Age")
plt.ylabel("Credit History to Age")
plt.show()
