#%%
import pandas as pd
import numpy as np
#%%
df = pd.read_csv('amazon_data_cleaned_01.csv')
df.columns
# %%
df.drop('Unnamed: 0', axis=1,inplace=True)
# %%
#Cleaning number of reviews
pd.set_option('display.max_colwidth', 30)
df['number_reviews'] = df['number_reviews'].str.replace('ratings','')
df['number_reviews'] = df['number_reviews'].str.replace('Nan','0')
df['number_reviews'] = df['number_reviews'].str.replace(',','')
df['number_reviews'] = df['number_reviews'].str.replace('rating','').astype('int')
# %%
#cleaning rating out of 5
df['rating'] = df['rating'].str.replace('out of 5','').astype('float')
#%%
#cleaning weight and its dimentions ['pounds', nan, 'ounces', 'grams', 'kilograms']
#divide weight into float and dimention
df['weight_float'] = df['weight'].apply(lambda x:  x if type(x)==float else x.split()[0]).astype(float)
df['weight_dimention'] = df['weight'].apply(lambda x:  x if type(x)==float else x.split()[1]).str.lower()

# %%


class dimention_convertion():
    '''
     pounds' 'ounces', 'grams' to 'kilograms'
    '''
    def __init__(self,value_float, dimention) :
        self.value = value_float
        self.dimention = dimention
        
    def convert (self):
        if self.dimention ==  'pounds':
            self.from_pound_to_kg() 
            return self.value
        elif self.dimention ==  'ounces':
            self.from_ounces_kg()
            return self.value
        elif self.dimention == 'grams':
            self.from_grams_to_kg()
            return self.value
        elif self.dimention == 'kilograms':
            return self.value

    def from_pound_to_kg( self):
       
           self.value =  self.value / 2.20462
       
    def from_ounces_kg(self):
       
            self.value =  self.value / 35.247
        
    def from_grams_to_kg(self):
        
           self.value =  self.value / 1000
#apply convertion_dimention class
df['weight_kg'] = df.apply(lambda row: dimention_convertion(row.weight_float ,row.weight_dimention).convert(), axis=1)
df.drop(['weight_float','weight_dimention','weight','mat_url', 'name','table_features_color_care_material'], axis=1, inplace=True)
# %%
df.head()
#%%
df.shape
# %%
#Converting dimention in volume
df[]
#%%
#  'number_reviews', 'rating',
#        'reviews_text', 'other_prices',
#        'other_colors', 'combined_price', 'weight',
#  'dimentions', 'care',
#   'thickness', 'material', 'brand', 'color'
# %%
