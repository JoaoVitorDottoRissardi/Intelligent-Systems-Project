import pandas as pd
import joblib

choosen_models = []

data = pd.read_csv('id3_trees.csv')

column = data['accuracy']

n_distinct_values = column.nunique()

distinct_values = sorted(column.unique().tolist())

selected_values = []

selected_values.append(distinct_values[int(n_distinct_values/4)])
selected_values.append(distinct_values[int(n_distinct_values/2)])
selected_values.append(distinct_values[int(3*n_distinct_values/4)])
selected_values.append(distinct_values[n_distinct_values-1])

print(selected_values)

print("\n First Selection: ")

first_selection = data[data['accuracy'] == selected_values[0]]

first_max_f_measure = first_selection['average f_measure'].max()

first_selection = first_selection[first_selection['average f_measure'] == first_max_f_measure]

print(first_selection[['model_index', 'random_state', 'max_depth', 'min_samples_leaf','criterion','accuracy', 'average f_measure']])

choosen_models.append(first_selection.loc[:, ['model_index','random_state', 'max_depth', 'min_samples_leaf','criterion']].head(1))

print("\n Second Selection: ")

second_selection = data[data['accuracy'] == selected_values[1]]

second_max_f_measure = second_selection['average f_measure'].max()

second_selection = second_selection[second_selection['average f_measure'] == second_max_f_measure]

print(second_selection[['model_index', 'random_state', 'max_depth', 'min_samples_leaf','criterion','accuracy', 'average f_measure']])

choosen_models.append(second_selection.loc[:, ['model_index','random_state', 'max_depth', 'min_samples_leaf','criterion']].head(1))

print("\n Third Selection: ")

third_selection = data[data['accuracy'] == selected_values[2]]

third_max_f_measure = third_selection['average f_measure'].max()

third_selection = third_selection[third_selection['average f_measure'] == third_max_f_measure]

print(third_selection[['model_index', 'random_state', 'max_depth', 'min_samples_leaf','criterion','accuracy', 'average f_measure']])

choosen_models.append(third_selection.loc[:, ['model_index','random_state', 'max_depth', 'min_samples_leaf','criterion']].head(1))

print("\n Fourth Selection: ")

fourth_selection = data[data['accuracy'] == selected_values[3]]

fourth_max_f_measure = fourth_selection['average f_measure'].max()

fourth_selection = fourth_selection[fourth_selection['average f_measure'] == fourth_max_f_measure]

print(fourth_selection[['model_index', 'random_state', 'max_depth', 'min_samples_leaf','criterion','accuracy', 'average f_measure']])

choosen_models.append(fourth_selection.loc[:, ['model_index','random_state', 'max_depth', 'min_samples_leaf','criterion']].head(1))

joblib.dump(choosen_models, 'id3_models.pkl')
