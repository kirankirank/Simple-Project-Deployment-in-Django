

from django.shortcuts import render
import joblib
import pickle


def home(request):
    return render(request,'index.html')
def result(request):
    
    
    M = pickle.load(open("mlr.pkl", "rb"))
    ok = joblib.load('ordinalEnc')
   
    eng = request.GET["eng"]
    hp = int(request.GET["hp"])
    vol = int(request.GET["vol"])
    sp = int(request.GET["sp"])
    wt = int(request.GET["wt"])
    print(eng,hp,vol,sp,wt)
    predictions=M.predict(ok.transform([[eng, hp, vol, sp, wt]]))
    print(predictions)
    constant={
        'y':predictions[0]
        }
    
    return render(request,"index.html" , constant)