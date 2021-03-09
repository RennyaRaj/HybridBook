import streamlit as st
import numpy as np
import pandas as pd

import sqlite3
conn=sqlite3.connect('data.db')
c=conn.cursor()

import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow.keras as tf
import joblib

import base64
from io import BytesIO

import bz2
import pickle
import _pickle as cPickle

from streamlit import caching


# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
  with bz2.BZ2File(title + '.pbz2','w') as f:
    cPickle.dump(data, f)

# Load any compressed pickle file
def decompress_pickle(file):
  data = bz2.BZ2File(file, 'rb')
  data = cPickle.load(data)
  return data



ratings_data = decompress_pickle('rat1.pbz2') 

ratings_df1=ratings_data.sort_values(by="user_id",ascending=True).reset_index(drop=True)

ratings_df=ratings_df1[ratings_df1["user_id"]<2501].reset_index(drop=True)

del ratings_data,ratings_df1




new_model=tf.models.load_model("modelrecsys.h5")
co=joblib.load("contentsfile.joblib")
titlefile=joblib.load('title.joblib')


####To download dataframe recommondations
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    #Generates a link allowing the data in a given panda dataframe to be downloaded
    #in:  dataframe
    #out: href string
    
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download csv file</a>' # decode b'abc' => abc

##df = ... # your dataframe
##st.markdown(get_table_download_link(df), unsafe_allow_html=True)




def create_usertable():
  c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')

def add_userdata(username,password):
  c.execute('INSERT INTO userstable(username, password) VALUES(?,?)',(username,password))
  conn.commit()

def login_user(username,password):
  c.execute('SELECT * FROM userstable WHERE username=? AND password=?',(username,password))
  data=c.fetchall()
  return data

def view_all_users():
  c.execute('SELECT * FROM userstable')
  data=c.fetchall()
  return data


st.title("      GOODREADS BOOKS   ")
st.markdown("###***HYBRID BOOK RECOMMENDATION SYSTEM USING DEEP LEARNING***")


menu=["Home", "Sign up", "Login","Books"]
choice=st.sidebar.selectbox("Menu",menu)

if choice=="Home":
  st.image("https://images.gr-assets.com/misc/1397605627-1397605627_goodreads_misc.png",width=850)

  st.markdown("Please use sidebar dropdown benu for ***Login/Signup***. (Login after Signing up entering custom username and password and tick login)")
  st.markdown("_you are in HOME view_")

elif choice=="Login":
  st.subheader("Login Section")
  
  username=st.sidebar.text_input("username")
  password=st.sidebar.text_input("password",type='password')
  
  if st.sidebar.checkbox("Login"):
    
    # if password=="12345":
    create_usertable()
    result=login_user(username,password)
    if result:

      st.success("LOGGED IN SUCCESSFULLY AS {} ".format(username))
      st.markdown("After Login please select any one of below Task options -**_Info_, _Start-Analytics_ (for Reccomondations), _Account Details_")
      
      
      
      task=st.selectbox("Task",["Info","Start-Analytics","Account Details"])
      
      if task=="Info":
        st.subheader("use Start-Analytics for finding Reccomondations")
        st.image("http://knowledgequest.aasl.org/wp-content/uploads/2019/05/GoodReads-logo.jpg",width=500)
        st.markdown("""**What is Goodreads?**

Goodreads is the largest site for connecting readers with books they will love. Keep track of books you have read, want to read, and are currently reading. Connect with other readers, leave reviews, and get book recommendations. You can even follow your favorite authors! (Follow me on goodreads)

Goodreads has been around since 2006, but it gained momentum in 2013 when it was bought by Amazon. Now you can integrate goodreads with your kindle and read book samples from Amazon straight from the goodreads site!

When most people think of goodreads, they think of book reviews and book recommendations. These are two of the most important features of goodreads, but there is so much more you can get from the website and app. Read on to learn how to use goodreads and take advantage of the lists, groups, giveaways, quotes, and so much more.""")
      
      elif task=="Start-Analytics":

        st.subheader("Top N number of Book Recommondations predicted realtime") 
        st.markdown("Please click enter key after entering values to apply") 

        
        #user_id = st.number_input('user_id',  min_value=1, max_value=53424, value=1)
        
        user_id=st.text_input("Enter user_id {1-2500} default 1")
        
        if user_id!="":
            user_id=int(user_id)
            if user_id<1 or user_id>2500:
              
              user_id=1                
                
        else:
            user_id=1
            
        num=st.text_input("Enter required_reccomondation_count (2-30) default 2")


        if num!="":
            num=int(num)
            if num<2 or num>30:
                num=2                

        else:
            num=2

        us_id_temp=[user_id for i in range(len(co['book_id']))]
        
        @st.cache(suppress_st_warning=True)
        def pred(new_model,us_id_temp,co,ratings_df,user_id,titlefile):
          reccom = new_model.predict([pd.Series(us_id_temp),co['book_id'],co.iloc[:,1:]])
          recc_df=pd.DataFrame(reccom,columns=["rating"])
          recc_df["book_id"]=co['book_id'].values


          df_new=ratings_df.where(ratings_df["user_id"]==user_id)
          df_new.dropna(inplace=True)
          list_books_seen=df_new['book_id'].tolist()
          del df_new

          recc_df_table = recc_df[~recc_df.book_id.isin(list_books_seen)]
          recc_df.sort_values(by="rating",ascending=False,inplace=True)   
          recc_df=recc_df.iloc[6:36].reset_index(drop=True)

          #num= st.number_input('required_reccomondation_count',  min_value=2, max_value=30, value=5)

          

          recc_df_table =recc_df.iloc[:num]
          recc_df_table=pd.merge(recc_df_table,titlefile,left_on="book_id",right_on="book_id")


          recc_df_table_new = recc_df_table.iloc[:,:6].reset_index(drop=True)

          st.write(recc_df_table_new)


          st.markdown(get_table_download_link(recc_df_table_new), unsafe_allow_html=True)
          for i in range(len(recc_df_table_new.index)):
            st.image( recc_df_table.iloc[i,7],
                  width=200, # Manually Adjust the width of the image as per requirement
              caption=recc_df_table.iloc[i,4]
              )
            
        if st.button("Reccomend"):
          caching.clear_cache()
          st.success("Showing {} Recommondations for user_id {} ".format(num,user_id))
          pred(new_model,us_id_temp,co,ratings_df,user_id,titlefile)


        
      elif task=="Account Details":
        st.subheader("User Profiles")
        user_result=view_all_users()
        clean_db=pd.DataFrame(user_result,columns=["Username","Password"])
        st.dataframe(clean_db)
        
      

    else:
      st.warning("Incorrect password/username")

elif choice=="Sign up":
  st.subheader("Create New Account")
  newuser=st.sidebar.text_input("username")
  newpassword=st.sidebar.text_input("password",type='password')

  if st.button("Sign up"):
    create_usertable()
    add_userdata(newuser,newpassword)
    st.success("You have successfully created a valid account")
    st.info("Goto Login menu")

elif choice=="Books":
  st.subheader("Books Present")
  st.markdown("Showing 8900 books of different languages")
  st.write(titlefile.iloc[:,:6])
  st.markdown(get_table_download_link(titlefile.iloc[:,:6]), unsafe_allow_html=True)
  
  
