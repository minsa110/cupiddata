# %%
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# %%
data = pd.read_parquet("parsed_data_public.parquet")
question_data = pd.read_csv("question_data.csv", sep=";")
test_items = pd.read_csv("test_items.csv") # cognititive ability test questions

# %%
data.d_age.describe()

# %%
max_age = data.d_age.min()
min_age = data.d_age.max()

# %%
data.d_age.plot.hist(bins=25, alpha=0.5)

# %%
age = data.d_age.tolist()

# %%
sns.histplot(data.d_age, kde=True, bins=25)
plt.xlabel('Age')
plt.axvline(data.d_age.mean(), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(data.d_age.mean()*1.1, max_ylim*0.9,
         'Mean: {:.2f}'.format(data.d_age.mean()))
plt.show()

# %%
data.d_religion_seriosity.value_counts()

## Cognitive ability test results

# %%
ca_test = data.copy()

right_answers = []
for ID, ROW in test_items.iterrows():
    right_answers.append(ROW.iloc[ROW["option_correct"] + 2])
test_items["right_answer"] = right_answers

for ID, ROW in test_items.iterrows():
    QUESTION = "q" + str(ROW["ID"])
    ANSWER = str(ROW["right_answer"])
    try:
        ca_test.dropna(subset=[QUESTION], inplace=True)
        ca_test["resp_" + QUESTION] = ca_test.apply(lambda row: row[QUESTION] == ANSWER, axis=1)
    except KeyError:
        print(f"{QUESTION} not found.")

# fix integer results that were stored as strings
ca_test.q18154 = pd.Series(ca_test.q18154, dtype="int")
ca_test.q18154 = pd.Series(ca_test.q18154, dtype="string")
ca_test.resp_q18154 = ca_test.apply(lambda row: row["q18154"] == "26", axis=1)

ca_test.q255 = pd.Series(ca_test.q255, dtype="int")
ca_test.q255 = pd.Series(ca_test.q255, dtype="string")
ca_test.resp_q255 = ca_test.apply(lambda row: row["q255"] == "89547", axis=1)

cognitive_score = ca_test[list(ca_test.filter(regex="^resp"))].sum(axis=1)
ca_test["cognitive_score"] = cognitive_score

palette = sns.color_palette("husl")
sns.palplot(palette)

sns.histplot(ca_test.cognitive_score, kde=True, bins=6)
sns.set_palette(palette)

## Relate cognitive ability to being dog / cat person

# %%
sns.catplot(x=ca_test.q997, y=ca_test.cognitive_score,
kind="box", height=5, aspect=2, data=ca_test).set_axis_labels("q997 = Are you a cat person or a dog person?", "cognitive score")
sns.set_palette(palette)

# %%
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x=ca_test.q997, y=ca_test.cognitive_score, ax=ax, data=ca_test)
sns.set_palette(palette)
plt.xlabel("q997 = Are you a cat person or a dog person?")
plt.ylabel("cognitive score")

## Anova test

# %%
dog_or_cat_pivot = ca_test.pivot(columns="q997", values="cognitive_score")
dog_or_cat_pivot.drop(dog_or_cat_pivot.columns[0], axis=1, inplace=True)
dog_or_cat_samples = [dog_or_cat_pivot[col].dropna() for col in dog_or_cat_pivot]

# %%
num_groups = len(dog_or_cat_pivot.columns)
num_observations = len(dog_or_cat_pivot)
dfn = num_groups - 1
dfd = num_observations - num_groups

f_critical = stats.f.ppf(q=0.95, dfn=dfn, dfd=dfd)

f_value, p_value = stats.f_oneway(*dog_or_cat_samples)

# %% [markdown]

# The value of f_critical (2.62) is larger than that of f_value (0.58). This means that the variance between the means of these groups is not significantly different. 
# The p value (0.63) is well over 0.05, which gives support to the null hypothesis (detracting support from the hypothesis that there is a positive correlation between someone's cognitive ability and whether they're a dog or cat person)
