#%%
import pandas as pd
#%%
df = pd.read_csv('amazon_data_cleaned_01.csv')
df.columns
# %%
df.drop('Unnamed: 0', axis=1,inplace=True)
# %%

# 'mat_url', 'name', 'number_reviews', 'rating',
#        'reviews_text', 'table_features_color_care_material', 'other_prices',
#        'other_colors', 'combined_price', 'weight', 'dimentions', 'care',
#        'thickness', 'material', 'brand', 'color'
#Cleaning number of reviews
pd.set_option('display.max_colwidth', None)
df['number_reviews'] = df['number_reviews'].str.replace('ratings','')
df['number_reviews'] = df['number_reviews'].str.replace('Nan','0')
df['number_reviews'] = df['number_reviews'].str.replace(',','')
df['number_reviews'] = df['number_reviews'].str.replace('rating','').astype('int')
df['other_prices'].unique()
# %%
#cleaning rating out of 5
df['rating'] = df['rating'].str.replace('out of 5','').astype('float')
#%%
#cleaning other prices

# %%
