#%%
import pandas as pd
import numpy as np
import re
#%%
df = pd.read_csv('amazon_data_cleaned_01.csv')
df.columns
df = df.replace('nan', np.nan, regex=True)
df = df.replace(0, np.nan, regex=True)

  
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
df.drop(['weight_float','weight_dimention','weight','mat_url','table_features_color_care_material'], axis=1, inplace=True)
# %%
df.head()
#%%
df.shape

# %%
#Converting dimentions and thickness in three rows: L X W X thickness_two
#L
df['dimentions'] = df['dimentions'].str.strip().str.lower()
df['L'] = df['dimentions'].apply(lambda x:  x if type(x)==float else x.split('x',1)[0]).astype(float)

#W
df['W'] = df['dimentions'].apply(lambda x:  x if type(x)==float else x.split('x',1)[1]).str.strip()
df['W'] = df['W'].apply(lambda x:  x if type(x)==float else x.split(' ',1)[0]).astype(float)

#thickness
#select from W forward
df['thickness_2'] = df['dimentions'].apply(lambda x:  x if type(x)==float else x.split('x',1)[1]).str.strip()
#select from thickness forward
df['thickness_2'] = df['thickness_2'].apply(lambda x:  x if type(x)==float else x.split(' ',1)[1]).str.strip()

#select only thickness
regex_thickness = re.compile(r'[\d+\.]+[\d*]')
df['thickness_2']= df['thickness_2'].apply(lambda x:  x if type(x)==float else re.findall(regex_thickness,x))
df['thickness_2'] = df['thickness_2'].apply(lambda y: np.nan  if y==[] else float(str(y).replace("['",'').replace("']",'')))
df['thickness_2'][df['thickness_2'].notnull()]
#dimention
#select only text
regex_dimention = re.compile(r'[a-z]+')
df['dimention_LXW']= df['dimentions'].apply(lambda x:  x if type(x)==float else re.findall(regex_dimention,x))
df['dimention_LXW'] = df['dimention_LXW'].apply(lambda y: np.nan  if y==[]  else str(y))
df['dimention_LXW'].unique()
#all is in inches 


#%%

# %%

class dimention_convertion_thickness():
    '''
     'feet', 'micron' 'centimeters', 'inches' to 'milimeters'
    '''
    def __init__(self,value_float, dimention) :
        self.value = value_float
        self.dimention = dimention
        
    def convert (self):
        if self.dimention == 'feet':
            self.from_feet_to_mm() 
            return self.value
        elif self.dimention ==  'micron':
            self.from_micron_to_mm()
            return self.value
        elif self.dimention == 'centimeters':
            self.from_cm_to_mm()
            return self.value
        elif self.dimention == 'inches':
            self.from_inches_to_mm()
            return self.value
        elif self.dimention == 'milimeters':
            return self.value
        elif self.dimention == 'mils':
            return self.value    

    def from_feet_to_mm( self):
       
           self.value =  self.value * 304.8
       
    def from_micron_to_mm(self):
       
            self.value =  self.value / 1000
        
    def from_cm_to_mm(self):
        
           self.value =  self.value * 100

    def from_inches_to_mm(self):
        
           self.value =  self.value * 25.4


df['thickness']=df['thickness'].str.strip()
df['thickness_float'] = df['thickness'].apply(lambda x:  x if type(x)==float else x.split()[0]).astype(float)
df['thickness_dim']= df['thickness'].apply(lambda x:  x if type(x)==float else x.split()[1]).str.lower().str.replace('item','').str.replace('league','')
          

#apply convertion_dimention class
df['thickness_mm'] = df.apply(lambda row: dimention_convertion_thickness(row.thickness_float ,row.thickness_dim).convert(), axis=1)
df.drop(['thickness','thickness_float','thickness_dim'], axis=1, inplace=True)
#%%
df.thickness_mm.unique()
#%%
#df.isna().sum()
#df['rating'].unique()
#  'number_reviews', 'rating',
#        'reviews_text', 'other_prices',
#        'other_colors', 'combined_price', 'weight',
#  'dimentions', 'care',
#   'thickness', 'material', 'brand', 'color'

