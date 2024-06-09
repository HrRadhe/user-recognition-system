from django.shortcuts import render,redirect
from .utils import check
import os
from django.conf import settings

def home(request):
    return render(request, 'FindFriend.html')

def result(request):
    context = {
        
    }
    if request.method == 'POST':
        data = {
        # 'First Name': None,
        # 'Surname' : None,
        # 'Contect' : None,
        # 'State' : None,
        # 'Current City' : None,
        # 'Native' : None,
        # 'Sex' : None
        }

        img = request.POST['image']
        Data = check(img)
        print(Data)
        j = 1
        for i in Data:
            data[j] = Data[i].iloc[0]
            j += 1
        context['data'] = data
        
        folder = int(Data.index[0])+1
        # print(folder)
        # print(settings.MEDIA_ROOT)
        folder_path = os.path.join(settings.MEDIA_ROOT, str(folder))
        # print(folder_path)

        image_urls = []
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                image_urls.append(f'{settings.MEDIA_URL}{folder}/{filename}')
        print(image_urls)
        context['images'] = image_urls
       
        
            
    return render(request, 'Result.html', context)