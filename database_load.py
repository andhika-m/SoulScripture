import sqlite3
import pandas as pd

def load_hadist_data():
  hadith_db_path = './database/hadist_database.db'
  conn = sqlite3.connect(hadith_db_path)
  hadist_tables = ['musnad_ahmad', 'musnad_darimi', 'musnad_syafii', 'muwatho_malik', 'shahih_bukhari', 
                 'shahih_muslim', 'sunan_abu_daud', 'sunan_ibnu_majah', 'sunan_nasai', 'sunan_tirmidzi']
  
  all_hadist_df = pd.concat([pd.read_sql_query(f"SELECT * FROM {table}", conn) for table in hadist_tables])
  conn.close()
  return all_hadist_df

def load_quran_data():
  conn = sqlite3.connect('./database/quran_database.db')
  ayat_df = pd.read_sql_query("SELECT * FROM table_ayat", conn)
  surah_df = pd.read_sql_query("SELECT * FROM table_surah", conn)
  conn.close()
  return ayat_df, surah_df