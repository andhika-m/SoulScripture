import pandas as pd
import re

def clean_text(text):
  text = re.sub(r'\[(.*?)\]', r'\1', text) # buat clean [] dalam sanad
  text = re.sub(r'_', ' ', text) # clean _ 
  text = re.sub(r'\s+', ' ', text) # ilangin spasi berlebih
  return text.strip().lower()

def preprocess_hadist(df):
  df['arab_clean'] = df['arab'].apply(clean_text)
  df['terjemah_clean'] = df['terjemah'].apply(clean_text)
  df['kitab_clean'] = df['kitab'].apply(clean_text)
  return df

def preprocess_quran(ayat_df, surah_df):
  ayat_df['Arab_clean'] = ayat_df['Arab'].apply(clean_text)
  ayat_df['Terjemahan_clean'] = ayat_df['Terjemahan'].apply(clean_text)
  
  # Gabungkan informasi surah ke dalam ayat
  ayat_df = pd.merge(ayat_df, surah_df[['Surah', 'Ayat']], on='Surah', how='left')
  ayat_df['Surah_Ayat'] = ayat_df['Ayat_y'] + ' (' + ayat_df['Surah'].astype(str) + ':' + ayat_df['Ayat_x'].astype(str) + ')'
    
  return ayat_df
