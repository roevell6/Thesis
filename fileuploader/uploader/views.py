from django.shortcuts import render
from django.core.files.storage import default_storage

def upload_file(request):
    if request.method == 'POST' and request.FILES.getlist('files'):  # Use getlist to handle multiple files
        file_names = []
        for file in request.FILES.getlist('files'):  # Loop through each file
            file_name = default_storage.save(file.name, file)
            file_names.append(file_name)
        return render(request, 'uploader/success.html', {'file_names': file_names})  # Pass file names to the template
    return render(request, 'uploader/upload.html')