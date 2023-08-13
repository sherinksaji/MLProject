#!/usr/bin/env python
# coding: utf-8

# ## Part 1

# In[256]:


# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt 


def file_to_df (file_path):
  with open(file_path, 'r',encoding='utf-8') as file:
      lines = file.readlines()
      data=[]
      for line in lines:
         if line.strip()!="":
          ls_x_y=line.strip().split(' ')
          y_value=ls_x_y.pop(-1)
          x_value=" ".join(ls_x_y)
          x_y_pair=[]
          x_y_pair.append(x_value)
          x_y_pair.append(y_value)
          data.append(x_y_pair)
      #data = [line.strip().split(' ', maxsplit=2)[:2] for line in lines]

  # print (data)
  # Create the dataframe
  df = pd.DataFrame(data, columns=['x', 'y'])
  #drop rows where value for y=None
  df = df.dropna(subset=['y'])

  df['x'] = df['x'].astype(str)
  df['y'] = df['y'].astype(str)
  # Display the dataframe
  #print("Here is dataframe created for the file path: ",file_path,"\n")
  #print(df.to_string)
  return df

test_line=". ... ... O"

ls_x_y=test_line.strip().split(' ')
# print(ls_x_y)
y_value=ls_x_y.pop(-1)
# print(y_value)
x_value=" ".join(ls_x_y)
# print(x_value)
x_y_pair=[]
x_y_pair.append(x_value)
x_y_pair.append(y_value)
# print(x_y_pair)

test_line=". ... ... O"
word, tag = test_line.strip().split(maxsplit=1)
# print("word:",word)
# print("tag:",tag)

file_path_train_es='Data/ES/train'
df_train_es=file_to_df(file_path_train_es)
# display(df_train_es)

file_path_train_ru='Data/RU/train'
df_train_ru=file_to_df(file_path_train_ru)
# display(df_train_ru)



# In[257]:


def file_to_df_dev_in (file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
      lines = file.readlines()
      data=[]
      for line in lines:
         if line.strip()!="":
          x_value=line.strip()
          data.append(x_value)
    df = pd.DataFrame(data, columns=['x'])
    return df

file_path_dev_in_ru="Data/RU/dev.in"
df_dev_in_ru= file_to_df_dev_in (file_path_dev_in_ru)
# display(df_dev_in_ru)


# In[258]:


def count_y(df,y_value):
    unique_counts = df['y'].value_counts().to_dict()
    # print("unique counts:",unique_counts)
    return unique_counts[y_value]


# In[259]:


def y_star_with_smallest_count(df):
     unique_counts = df['y_star'].value_counts().to_dict()
    #  print("unique_counts from y_star_with_smallest_count",unique_counts)
     min_count = min(unique_counts.values())
    #  print("min count",min_count)
     y_with_min_count = [key for key, value in unique_counts.items() if value == min_count]
    #  print("y with min count",y_with_min_count)         
     return y_with_min_count[0]         


# In[260]:


def count_of_y_with_smallest_count(df):
     unique_counts = df['y'].value_counts().to_dict()
     min_count = min(unique_counts.values())
     y_with_min_count = [key for key, value in unique_counts.items() if value == min_count]         
     return min_count 


# In[261]:


def create_df_filtered_for_y_value(df,y_value):
  return df[df['y'] == y_value]


# In[262]:


def create_df_x_count_y_to_x(df):
    df_x_count_y_to_x = df['x'].value_counts().reset_index()

    df_x_count_y_to_x.columns = ['x', 'count_y_to_x']

    return df_x_count_y_to_x


# In[263]:


def create_ls_of_all_y_values(df):
    unique_values = df['y'].unique()
    return (unique_values)
# print(create_ls_of_all_y_values(df_train_es))
# print(create_ls_of_all_y_values(df_train_ru))


# In[264]:


def create_df_e_x_y_train(train_df,y_value):
   
    df_train_filtered_for_y = create_df_filtered_for_y_value(train_df,y_value)
    df_e_x_y_train=create_df_x_count_y_to_x(df_train_filtered_for_y)
    df_e_x_y_train['e(x|y)'] = df_e_x_y_train['count_y_to_x']/(count_y(train_df,y_value))
    df_e_x_y_train['count y']=count_y(train_df,y_value)
    df_e_x_y_train['y']=y_value
    return df_e_x_y_train


df_e_x_y_train_for_I_neutral=create_df_e_x_y_train(df_train_es,"B-positive")
# display(df_e_x_y_train_for_I_neutral)
# print(df_e_x_y_train_for_I_neutral['e(x|y)'].sum())



# In[265]:


def create_e_x_y_df_train_all_y_values(file_path):
    
  df_train=file_to_df(file_path)
  #print(df_train_es)
  ls_df_train=[]
  ls_y_values=create_ls_of_all_y_values(df_train)
  #print(len(ls_y_values))
  for y_value in ls_y_values:
    if y_value!=None:
      df_e_x_y=create_df_e_x_y_train(df_train,y_value)
      ls_df_train.append(df_e_x_y)
  #print(len(ls_df_train))
  
  combined_df_train=pd.concat(ls_df_train,axis=0)
  return (combined_df_train)


file_path_train_es='Data/ES/train'
print("Part 1 Question 1 answers - ES Dataset")
print(create_e_x_y_df_train_all_y_values(file_path_train_es))

file_path_train_ru='Data/RU/train'
print("Part 1 Question 1 answers - RU Dataset")
print(create_e_x_y_df_train_all_y_values(file_path_train_ru))



# In[266]:


def create_df_e_x_y_test_dev_in_all_y_values(file_path_dev_in,file_path_train):
    df_dev_in=file_to_df_dev_in (file_path_dev_in)
    df_train_with_e_x_y=create_e_x_y_df_train_all_y_values(file_path_train)
    k=1
    

    # Merge the DataFrames based on the 'x' column
    merged_df = df_dev_in.merge(df_train_with_e_x_y, on='x', how='left')
  

    y_values=create_ls_of_all_y_values(df_train_with_e_x_y)
    new_rows=[]
    nan_rows = merged_df[merged_df['y'].isna()]
    for _, row in nan_rows.iterrows():
        x_value = row['x']
        for y_value in y_values:
            #numerators of e(x|y) are now already taken care of
            new_rows.append({'x': "#UNK#", 'count_y_to_x':k,'y': y_value, 'count y':count_y(df_train_with_e_x_y,y_value)})

    df_of_new_rows = pd.DataFrame(new_rows)
    
    df_dev_in_with_e_x_y = pd.concat([merged_df[merged_df['y'].notna()], df_of_new_rows], ignore_index=True)
    #now the denominator in e(x|y) is taken care of
    df_dev_in_with_e_x_y ["count_y_plus_k"]=df_dev_in_with_e_x_y["count y"]+k
    #now just do divide count y_to_x by count_y_plus_k
    df_dev_in_with_e_x_y ["e(x|y)"]=df_dev_in_with_e_x_y ["count_y_to_x"]/df_dev_in_with_e_x_y ["count_y_plus_k"]
    return (df_dev_in_with_e_x_y)

file_path_dev_in_es="Data/ES/dev.in"
file_path_train_es="Data/ES/train"

file_path_dev_in_ru="Data/RU/dev.in"
file_path_train_ru="Data/RU/train"

print("Part 1 Question 2 answers - ES Data")
print(create_df_e_x_y_test_dev_in_all_y_values(file_path_dev_in_es,file_path_train_es))
df_dev_in_es_test_e_x_y=create_df_e_x_y_test_dev_in_all_y_values(file_path_dev_in_es,file_path_train_es)
csv_dev_in_es_test_e_x_y=df_dev_in_es_test_e_x_y.to_csv('csv_dev_in_es_test_e_x_y.csv',index=True)

print("Part 1 Question 2 answers - RU Data")
print(create_df_e_x_y_test_dev_in_all_y_values(file_path_dev_in_ru,file_path_train_ru))

df_dev_in_ru_test_e_x_y=create_df_e_x_y_test_dev_in_all_y_values(file_path_dev_in_ru,file_path_train_ru)
csv_dev_in_ru_test_e_x_y=df_dev_in_ru_test_e_x_y.to_csv('csv_dev_in_ru_test_e_x_y.csv',index=True)


# In[267]:


def create_df_x_to_y_star_with_train_and_dev_in(file_path_dev_in,file_path_train):
    e_x_y_df_dev_in=create_df_e_x_y_test_dev_in_all_y_values(file_path_dev_in,file_path_train)
    #display(e_x_y_df_dev_in)
    # Group by 'x' and find the maximum 'e(x|y)' value for each group
    df_x_to_y_star = e_x_y_df_dev_in.groupby('x').apply(lambda group: group.loc[group['e(x|y)'].idxmax()]).reset_index(drop=True)
  

    return df_x_to_y_star

   

file_path_dev_in_es="Data/ES/dev.in"
file_path_train_es="Data/ES/train"

file_path_dev_in_ru="Data/RU/dev.in"
file_path_train_ru="Data/RU/train"

print("Part 1 Question 3 Answers - ES Data")
print(create_df_x_to_y_star_with_train_and_dev_in(file_path_dev_in_es,file_path_train_es))

print("Part 1 Question 3 Answers - RU Data")
print(create_df_x_to_y_star_with_train_and_dev_in(file_path_dev_in_ru,file_path_train_ru))
    


# In[268]:


def generate_y_values_with_dev_in_y_star(file_path_dev_in,file_path_train,file_path_dev_p1_out):
    df_dev_in_y_star=create_df_x_to_y_star_with_train_and_dev_in(file_path_dev_in,file_path_train)
    #display(df_dev_in_y_star)
    df_train=file_to_df(file_path_train)
    x_values=df_train['x'].tolist()
    #x_values_df_dev_in=df_dev_in_y_star['x'].tolist()
    #print("Plato" in x_values_df_dev_in)
    with open(file_path_dev_in, 'r',encoding='utf-8') as file:
      lines = file.readlines()
      for l in range(len(lines)):
        line=lines[l].strip()
        if line in x_values:
          # print(line)
          possible_y_values=df_dev_in_y_star[df_dev_in_y_star['x'] == line]['y'].tolist()
          lines[l]=line+" "+possible_y_values[0]
         
          if (len(possible_y_values)!=1):
            print ("something wrong: x_vlaues in df_dev_in_y_star not unique for some reason,for line: ",l,line,possible_y_values)
        else:
          if (line!=""):
            #  print("here")
             x_value="#UNK#"
             y_values=df_dev_in_y_star[df_dev_in_y_star['x'] == x_value]['y'].tolist()
             line="#UNK#"+" "+y_values[0]
             lines[l]=line
          else:
            lines[l]=""
          
    with open(file_path_dev_p1_out, 'w',encoding='utf-8') as file:
        
        for line in lines:
          file.write(line+"\n")      

file_path_dev_in_es = 'Data/ES/dev.in'   
file_path_train_es='Data/ES/train'  
file_path_dev_p1_out_es='Data/ES/dev.p1.out' 
generate_y_values_with_dev_in_y_star(file_path_dev_in_es,file_path_train_es,file_path_dev_p1_out_es)


file_path_dev_in_ru = 'Data/RU/dev.in'
file_path_train_ru ='Data/RU/train'
file_path_dev_p1_out_ru='Data/RU/dev.p1.out'
generate_y_values_with_dev_in_y_star(file_path_dev_in_ru,file_path_train_ru,file_path_dev_p1_out_ru)

