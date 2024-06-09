from data.save_data import save_pickled_data
from data.get_data import get_json_data
from utils import train_models
from translate import translate_keys

# Load data
data = get_json_data('offers')

# Translate keys
data = list(map(translate_keys, data))

print('Training models...')
# Train models
df, model_pipelines, metrics_scores = train_models(data)

print('Models loaded!')
print('Saving dataframe...')

# Save model
save_pickled_data(df, 'dataframe')

print('Dataframe saved!')
print('Saving models...')

# Save each model separately
for name, model in model_pipelines.items():
    save_pickled_data(model, f'{name}_model')
    print(f'Model {name} saved!')

print('Models saved!')
print('Saving metrics scores...')

# Save MAE scores
save_pickled_data(metrics_scores, 'metrics_scores')

print('MAE scores saved!')