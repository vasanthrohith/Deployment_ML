from tkinter import *
from tkinter import ttk

import numpy as np
import pandas as pd
from tkinter import filedialog

# ML agos
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


from sklearn.preprocessing import StandardScaler



class base():
    def __init__(self):
        print("Base started")
        
        
    def base_fn(self):
        ob=home_page()
        ob.home_fn()
        
        
class home_page(base):
    global data, path
    
    def home_fn(self):
        global x_index,y_index,selected_x,selected_y
        x_index=[]
        y_index=[]
        selected_x=[]
        selected_y=[]
        
        
        def select_x_fn():
            
            # Here the selected one from the combobox will get and located the index num in dataset and the index num will be appended in list_index as X var
            x=select_col.get()
            # print("selected column = ",x)
            selected_x.append(x)
            
            ind_x=data.columns.get_loc(x)
            # print("X cols-------->",ind_x)
            x_index.append(ind_x)
            
            # inserting to the text box
            x_txt_box.insert(END,x+'\n')
            
            
        def select_y_fn():
            # Here the selected one from the combobox will get and located the index num in dataset and the index num will be appended in list_index as X var
            y=select_col.get()
            selected_y.append(y)
            # print("selected column = ",y)
            ind_y=data.columns.get_loc(y)
            # print("Y cols-------->",ind_y)
            y_index.append(ind_y)
            y_txt_box.insert(END,y+'\n')
            
        def page_2_fn():
            
            # this fn to 
            # print(list_index)
            x_col=x_index
            y_col=y_index
            name_x=selected_x
            name_y=selected_y
            win.destroy()
            obj=algorithm_choose()
            obj.algorithms(path, x_col, y_col,name_x,name_y)
            
            
    
            
            
        def open_file_fn():
            global path, data
            file=filedialog.askopenfilename(title="Open File",filetypes=(("csv files","*.csv"),("Text Files","*.txt"),("All files","*."),("Python files","*.py")))
            # print(file)
            path=file.replace("/","//")
            # print(path)
            data=pd.read_csv(path)
            list_1=list(data.columns)
            # print(list_1)
            select_col['values'] = tuple(list_1)
            
            
            
        def refresh_fn():
            x_txt_box.delete('1.0',END)
            y_txt_box.delete('1.0',END)
            y_index.clear()
            x_index.clear()
            selected_x.clear()
            selected_y.clear()
        
        
        
        
        
        
        win=Tk()
        win.geometry('500x500')
        win.title("Prediction Algorithms")
        
        frame1=Frame(win,bg='purple')
        frame1.pack(ipadx=50,ipady=50,expand=True,fill='both')
        
        header_l=Label(frame1,text='PREDICTIONS',font=("Futura","28","bold"),bg='red',relief=RAISED,padx=6)
        header_l.pack(ipady=10)
        
        select_col_en=StringVar()
        # keepvalue=month_en.get()
        select_col=ttk.Combobox(frame1,width=30,textvariable=select_col_en)
        select_col.place(x=670,y=270)
        
         #-->to append the values in combobox
        
        select_x_btn=Button(frame1,text='select X',relief=GROOVE,command=select_x_fn)
        select_x_btn.place(x=670,y=350)
        
        select_y_btn=Button(frame1,text='select Y',relief=GROOVE,command=select_y_fn)
        select_y_btn.place(x=820,y=350)
        
        x_txt_box=Text(frame1,font=("Futura",'14',"bold"),width=30,height=10)
        x_txt_box.place(x=360,y=390)
        
        y_txt_box=Text(frame1,font=("Futura",'14',"bold"),width=30,height=10)
        y_txt_box.place(x=850,y=390)
        
        
        next_btn=Button(frame1,text='next',relief=GROOVE,command=page_2_fn)
        next_btn.place(x=750,y=620)
        
        refresh_btn=Button(frame1,text='refresh',relief=GROOVE,command=refresh_fn)
        refresh_btn.place(x=744,y=660)
        
        
        open_btn=Button(frame1,text='Open',relief=GROOVE,command=open_file_fn)
        open_btn.place(x=750,y=210)
        

        
        # month.current(12)
        
        
        
        win.mainloop()
        
        
        
        
        
class algorithm_choose(home_page):

    def algorithms(self,path, x_col, y_col,name_x,name_y):
        
        win1=Tk()
        win1.geometry('500x500')
        # win1.state('zoomed')
        win1.title("Prediction Algorithms")
        win1.config(background='black')
        frame2=Frame(win1,bg='purple')
        frame2.pack(ipadx=50,ipady=50,expand=True,fill='both')
        
        
        
        def slin_reg_fn():      #Simple linear regression
            
         
            
            frame2.destroy()
            
            header_l=Label(win1,text='Simple-Linear Regression',font=("Futura","28","bold"),bg='red',relief=RAISED,padx=6)
            header_l.pack(ipady=10)
            
            frame5=Frame(win1,bg='purple')
            frame5.pack(padx=190,pady=60,ipadx=50,ipady=50,expand=True,fill='both')
            
            
            
            
            def pred_slin_reg_fn():         #Simple linear regression model
                entry=col_1_e.get()
                # print(path, x_col, y_col,name_x,name_y)
                
                # Train splitting
                x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
                
                
                # Training model
                regressor=LinearRegression()
                regressor.fit(x_train,y_train)
                
                y_pred=regressor.predict(x_test)
                pred_y=regressor.predict([[entry]])
                print(pred_y)
                
                pred_l=Label(frame5,text=pred_y[0][0].round(2),font=("Futura","18","bold"),bg='light grey')
                pred_l.place(x=620,y=250)
                
                
                
                
                
            
            col_1_l=Label(frame5,text=f'{name_x[0]}',font=("Futura","18","bold"),bg='light grey')
            col_1_l.place(x=370,y=150)
            
            col_1_en=StringVar()
            col_1_e=Entry(frame5,width=20,textvariable=col_1_en)
            col_1_e.place(x=620,y=155)
            
            result_l=Label(frame5,text=f"{name_y[0]}",font=("Futura","18","bold"),bg='light grey')
            result_l.place(x=370,y=250)
            
            result_btn=Button(frame5,text='Predict',bg='light grey',relief=GROOVE,command=pred_slin_reg_fn)
            result_btn.place(x=510,y=355)
            
            
            
            
        def mlin_reg_fn():     #Multi linear regression
            frame2.destroy()
  
            
            header_l=Label(win1,text='Multi-Linear Regression',font=("Futura","28","bold"),bg='red',relief=RAISED,padx=6)
            header_l.pack(ipady=10)
            
            frame6=Frame(win1,bg='purple')
            frame6.pack(padx=190,pady=60,ipadx=50,ipady=50,expand=True,fill='both')
            
            
            def pred_mlin_reg_fn():  #Multi linear regression Model
                
                entry1=col_1_e.get()
                entry2=col_2_e.get()
                
                x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
                
                regressor=LinearRegression()
                regressor.fit(x_train,y_train)
                
                y_pred=regressor.predict(x_test)
                # print(y_pred)
                
                pred_y=regressor.predict([[entry1,entry2]])
                print(pred_y)
                
                pred_l=Label(frame6,text=pred_y[0][0].round(2),font=("Futura","18","bold"),bg='light grey')
                pred_l.place(x=600,y=300)
                
                
            
            col_1_l=Label(frame6,text=f'{name_x[0]}',font=("Futura","18","bold"),bg='light grey')
            col_1_l.place(x=350,y=150)
            
            col_2_l=Label(frame6,text=f'{name_x[1]}',font=("Futura","18","bold"),bg='light grey')
            col_2_l.place(x=350,y=210)
            
            col_3_l=Label(frame6,text=f'{name_y[0]}',font=("Futura","18","bold"),bg='light grey')
            col_3_l.place(x=350,y=300)
            
            
            col_1_en=StringVar()
            col_1_e=Entry(frame6,width=20,textvariable=col_1_en)
            col_1_e.place(x=600,y=155)
            
            col_2_en=StringVar()
            col_2_e=Entry(frame6,width=20,textvariable=col_2_en)
            col_2_e.place(x=600,y=210)
            
            
            result_btn=Button(frame6,text='Predict',bg='light grey',relief=GROOVE,command=pred_mlin_reg_fn)
            result_btn.place(x=460,y=375)
            
        
        def plin_reg_fn():
            
            frame2.destroy()
  
            
            header_l=Label(win1,text='Polynomial Regression',font=("Futura","28","bold"),bg='red',relief=RAISED,padx=6)
            header_l.pack(ipady=10)
            
            frame7=Frame(win1,bg='purple')
            frame7.pack(padx=190,pady=60,ipadx=50,ipady=50,expand=True,fill='both')
            
            def pred_poly_reg_fn():
                x_entry=col_1_e.get()
                poly_reg=PolynomialFeatures(degree=7)
                x1=poly_reg.fit_transform(x)
                
                
                regressor=LinearRegression()
                regressor.fit(x1,y)
                
                
                pred_y=regressor.predict(poly_reg.fit_transform([[x_entry]]))
                print(pred_y)
                
                pred_l=Label(frame7,text=pred_y[0][0].round(2),font=("Futura","18","bold"),bg='light grey')
                pred_l.place(x=600,y=250)
                
                
            
            col_1_l=Label(frame7,text=f'{name_x[0]}',font=("Futura","18","bold"),bg='light grey')
            col_1_l.place(x=350,y=150)
            
            col_1_en=StringVar()
            col_1_e=Entry(frame7,width=20,textvariable=col_1_en)
            col_1_e.place(x=600,y=155)
            
            result_l=Label(frame7,text=f"{name_y[0]}",font=("Futura","18","bold"),bg='light grey')
            result_l.place(x=350,y=250)
            
            result_btn=Button(frame7,text='Predict',bg='light grey',relief=GROOVE,command=pred_poly_reg_fn)
            result_btn.place(x=460,y=355)
        
        
        
        #Classification------------------
        
        
        def lcls_fn():
            
            frame2.destroy()
  
            
            header_l=Label(win1,text='Logistic Regression',font=("Futura","28","bold"),bg='red',relief=RAISED,padx=6)
            header_l.pack(ipady=10)
            
            frame8=Frame(win1,bg='purple')
            frame8.pack(padx=190,pady=60,ipadx=50,ipady=50,expand=True,fill='both')
            
            def pred_logcls_fn():
                x_entry1=col_1_e.get()
                x_entry2=col_2_e.get()
                
                
                
                sc=StandardScaler()
                x1=sc.fit_transform(x)
                x2=sc.fit_transform([[x_entry1,x_entry2]])   #fit transform user input
                
                
                # x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.30,random_state=0)
                
                
                
                classifier=LogisticRegression()
                classifier.fit(x1,np.ravel(y))
                
                pred_y=classifier.predict(x2)
                
                pred_l=Label(frame8,text=pred_y[0],font=("Futura","18","bold"),bg='light grey')
                pred_l.place(x=600,y=300)
                # print(pred_y)

                
            
            col_1_l=Label(frame8,text=f'{name_x[0]}',font=("Futura","18","bold"),bg='light grey')
            col_1_l.place(x=350,y=150)
            
            col_2_l=Label(frame8,text=f'{name_x[1]}',font=("Futura","18","bold"),bg='light grey')
            col_2_l.place(x=350,y=210)
            
            col_3_l=Label(frame8,text=f'{name_y[0]}',font=("Futura","18","bold"),bg='light grey')
            col_3_l.place(x=350,y=300)
            
            
            col_1_en=StringVar()
            col_1_e=Entry(frame8,width=20,textvariable=col_1_en)
            col_1_e.place(x=600,y=155)
            
            col_2_en=StringVar()
            col_2_e=Entry(frame8,width=20,textvariable=col_2_en)
            col_2_e.place(x=600,y=210)
            
            
            result_btn=Button(frame8,text='Predict',bg='light grey',relief=GROOVE,command=pred_logcls_fn)
            result_btn.place(x=460,y=375)
            
        
        
        def naive_fn():
            frame2.destroy()
  
            
            header_l=Label(win1,text='Naive Bayes',font=("Futura","28","bold"),bg='red',relief=RAISED,padx=6)
            header_l.pack(ipady=10)
            
            frame9=Frame(win1,bg='purple')
            frame9.pack(padx=190,pady=60,ipadx=50,ipady=50,expand=True,fill='both')
            
            
            def pred_nvbys_fn():
                x_entry1=col_1_e.get()
                x_entry2=col_2_e.get()
                
                classifier=GaussianNB()
                classifier.fit(x,y)
                
                
                
                pred_y=classifier.predict([[x_entry1,x_entry2]])
                print(pred_y)
                
                
                pred_l=Label(frame9,text=pred_y[0],font=("Futura","18","bold"),bg='light grey')
                pred_l.place(x=600,y=300)
                

            
        
            
            col_1_l=Label(frame9,text=f'{name_x[0]}',font=("Futura","18","bold"),bg='light grey')
            col_1_l.place(x=350,y=150)
            
            col_2_l=Label(frame9,text=f'{name_x[1]}',font=("Futura","18","bold"),bg='light grey')
            col_2_l.place(x=350,y=210)
            
            col_3_l=Label(frame9,text=f'{name_y[0]}',font=("Futura","18","bold"),bg='light grey')
            col_3_l.place(x=350,y=300)
            
            
            col_1_en=StringVar()
            col_1_e=Entry(frame9,width=20,textvariable=col_1_en)
            col_1_e.place(x=600,y=155)
            
            col_2_en=StringVar()
            col_2_e=Entry(frame9,width=20,textvariable=col_2_en)
            col_2_e.place(x=600,y=210)
            
            
            result_btn=Button(frame9,text='Predict',bg='light grey',relief=GROOVE,command=pred_nvbys_fn)
            result_btn.place(x=460,y=375)
        
        # print('algo class',path,"X : ",x_col,"............Y : ",y_col)
        dataset=pd.read_csv(path)
        # print(dataset)
        # print()
        x=dataset.iloc[:,x_col].values
        y=dataset.iloc[:,y_col].values
        # print("x : ",x)        
        # print("y : ",y)  
        
        
        header_l=Label(frame2,text='PREDICTION',font=("Futura","28","bold"),bg='red',relief=RAISED,padx=6)
        header_l.pack(ipady=10)
        
        frame3=Frame(frame2,bg='blue')
        frame3.pack(fill=BOTH,side=LEFT,expand=True,padx=10,pady=10)
        
        frame4=Frame(frame2,bg='red')
        frame4.pack(fill=BOTH,side=LEFT,expand=True,padx=10,pady=10)
        
        
        regression_l=Label(frame3,text='Regression Model',font=("Futura","22","bold"),bg='red',relief=RAISED,padx=6)
        regression_l.pack(pady=10)
        
        lin_reg_btn=Button(frame3,text="Simple-Linear Regression",font=("Futura","14","bold"),bg='light grey',relief=GROOVE,command=slin_reg_fn)
        lin_reg_btn.pack(pady=60)
        
        mul_reg_btn=Button(frame3,text="Multi-Linear Regression",font=("Futura","14","bold"),bg='light grey',relief=GROOVE,command=mlin_reg_fn)
        mul_reg_btn.pack(pady=80)
        
        pol_reg_btn=Button(frame3,text="Polynomial Regression",font=("Futura","14","bold"),bg='light grey',relief=GROOVE,command=plin_reg_fn)
        pol_reg_btn.pack(pady=100)
        
        
        
        classification_l=Label(frame4,text='Classification Model',font=("Futura","22","bold"),bg='red',relief=RAISED,padx=6)
        classification_l.pack(pady=10)
        
        log_cls=Button(frame4,text="Logistic Regression",font=("Futura","14","bold"),bg='light grey',relief=GROOVE,command=lcls_fn)
        log_cls.pack(pady=60)
        
        nav_bs_btn=Button(frame4,text="Naive - Bayes",font=("Futura","14","bold"),bg='light grey',relief=GROOVE,command=naive_fn)
        nav_bs_btn.pack(pady=80)
        
        # pol_reg_btn=Button(frame4,text="Polynomial Regression",bg='light grey',relief=RAISED,command=lin_reg_fn)
        # pol_reg_btn.pack(pady=100)
        
        
        
    
  
        
        
        
        win1.mainloop()
        
        
             
        
        
obb=base()
obb.base_fn()






























