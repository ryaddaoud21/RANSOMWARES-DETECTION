from msilib import datasizemask
import pickle
from django.http import HttpResponse
from django.shortcuts import render, redirect
import pandas as pd
from .forms import UserRegisterForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required



def home(request):
    return render(request, 'users/home.html')




def register(request):
    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Hi {username}, your account was created successfully !')
            return redirect('home')
    else:
        form = UserRegisterForm()

    return render(request, 'users/register.html', {'form': form})




@login_required()



def profile(request):
    return render(request, 'users/profile.html')



def start(request):
    print("Hi")

    return render(request, 'users/start.html')



def result(request):
    print("Bye")
    print(request.POST['Algo'])

    Result = ""



    File = request.FILES['file']

    data = pd.read_csv(File , index_col=False, sep=';')

    if 'Random Forest' in request.POST['Algo']:


      loaded_model = pickle.load(open('rf.sav', 'rb'))
      result = loaded_model.predict(data)

      if result == 0:
         Result = "not ransomware"
         print("not ransomware")
      elif result == 1:
         Result = "ransomware"

         print ("ransomware")
      else:

         Result = "error"

         print ("error")

    if 'Support Vector Machine' in request.POST['Algo']:
     
      print('SVM')
      loaded_model = pickle.load(open('svm.sav', 'rb'))
      result = loaded_model.predict(data)


      if result == 0:
         Result = "not ransomware"
         print("not ransomware")
      elif result == 1:
         Result = "ransomware"

         print ("ransomware")
      else:

         Result = "error"

         print ("error")



    if 'Multi-Layer Perception' in request.POST['Algo']:
    
     loaded_model = pickle.load(open('mlp.sav', 'rb'))
     result = loaded_model.predict(data)

     if result == 0:
         Result = "not ransomware"
         print("not ransomware")
     elif result == 1:
         Result = "ransomware"

         print ("ransomware")
     else:

         Result = "error"

         print ("error")

    if 'Decision Tree' in request.POST['Algo']:
    
     loaded_model = pickle.load(open('dtc.sav', 'rb'))

     result = loaded_model.score(data,[0])

     
     result = loaded_model.predict(data)

     
     if result == 0:
         Result = "not ransomware"
         print("not ransomware")
     elif result == 1:
         Result = "ransomware"

         print ("ransomware")
     else:

         Result = "error"

         print ("error")

    if 'Logistic Regression' in request.POST['Algo']:
    
     loaded_model = pickle.load(open('clfr.sav', 'rb'))
     result = loaded_model.predict(data)

     if result == 0:
         Result = "not ransomware"
         print("not ransomware")
     elif result == 1:
         Result = "ransomware"

         print ("ransomware")
     else:

         Result = "error"

         print ("error")

    if 'AdaBoost with Decision Tree' in request.POST['Algo']:
    
     loaded_model = pickle.load(open('AdeBoostDT.sav', 'rb'))
     result = loaded_model.predict(data)

     if result == 0:
         Result = "not ransomware"
         print("not ransomware")
     elif result == 1:
         Result = "ransomware"

         print ("ransomware")
     else:

         Result = "error"

         print ("error")

    context = { 'Result' : Result }

    return render(request, 'users/result.html',context)

    

    
    



    
