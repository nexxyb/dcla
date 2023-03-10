# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""


from urllib import request
from django import template, forms
from django.http import HttpResponse, HttpResponseServerError , HttpResponseBadRequest, JsonResponse
from django.template.response import TemplateResponse
from django.urls import reverse, reverse_lazy
from django.views import generic
from django.contrib.auth.mixins import LoginRequiredMixin
#from .models import Expense, Income, Project
from django.views.generic.edit import CreateView, DeleteView, UpdateView
from django.contrib.auth import get_user_model
from django.shortcuts import redirect, render
from django.core.mail import send_mail, BadHeaderError
import boto3
from datetime import timezone, datetime
from django.dispatch import receiver
from django.db.models.signals import post_save
from django.conf import settings
import pandas as pd
from .models import File
from . import cleaner
from django.core.files.storage import FileSystemStorage
from .forms import DataCleaningForm
from .tasks import clean_data
from celery.result import AsyncResult
import redis


User=get_user_model()
s3_client = boto3.resource('s3')

class IndexView(generic.TemplateView, LoginRequiredMixin ):
    template_name='home/raw2.html'
    
class FileUploadView(generic.TemplateView, LoginRequiredMixin ):
    
    def post(self, request):
        # Check if a file was uploaded
        if 'file' not in request.FILES:
            return HttpResponseBadRequest('No file was uploaded')
        
        # Get the uploaded file
        uploaded_file = request.FILES['file']
        
        # Validate the file
        if uploaded_file.size > 60 * 1024 * 1024:
            # File is larger than 10 MB
            return HttpResponseBadRequest('File is too large')
        
        if uploaded_file.content_type not in ['application/vnd.ms-excel', 'text/csv','text/tab-separated-values','application/json']:
            # File is not a valid image
            return HttpResponseBadRequest('Invalid file type')
        
        # Rename the file using the current timestamp
        file_name = str(datetime.now().timestamp())
        if settings.USE_S3:
            
        #key = settings.AWS_FILE_UPLOAD_LOCATION + file_name
        # Use the AWS SDK for Python (Boto3) to upload the file to the S3 bucket
            #try:
                #s3_client.meta.client.upload_file(uploaded_file, settings.AWS_STORAGE_BUCKET_NAME, key)
            
            # Save the uploaded file details to the database
            upload = File(
                            name=file_name,
                            size=uploaded_file.size,
                            content_type=uploaded_file.content_type,
                            upload_by=self.request.user,
                            file=uploaded_file
                                )
            upload.save()
            # except Exception as e:
            #     # Failed to upload the file to S3
            #     return HttpResponseBadRequest('Failed to upload file')
        
        else:
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
        return redirect(reverse('file-fetch', kwargs={'pk': upload.id}))

class FetchFileView(generic.DetailView, LoginRequiredMixin):
    model= File
    template_name='home/fetch.html'
    
    def get(self,  request, *args, **kwargs):
        # Call the parent class's get method to set the object attribute
        response = super().get(request, *args, **kwargs)

        
        # Access the query object
        pk = kwargs['pk']

        # Use the pk to retrieve the object from the database
        obj = File.objects.get(pk=pk)

        # Use get_context_data to create a context dictionary
        context = self.get_context_data(object=obj, **kwargs)

        datafile=obj.file
        # Load the file into a Pandas dataframe
        df = pd.read_csv(datafile)
        
        # Get the column names
        context['column_names'] = df.columns.to_list()
      
        # Store the data frame in the session
        request.session['data_frame'] = df.to_json()
        context['form']=DataCleaningForm()
        context['class_attrs'] = {'class': 'special'}
        # Do some processing on the dataframe
        #context['dimensions']= cleaner.calculate_dimensions(df),
        dimensions= cleaner.calculate_dimensions(df)
        context['rows']= dimensions[0]
        context['columns']= dimensions[1]
        context['total_nulls']=cleaner.calculate_null_values_total(df)
        context['null_percentage']= cleaner.calculate_null_percentage(df)
        context['preview']=df.head()
        context['size']=cleaner.dataframe_size(df)
        context['duplicates']=cleaner.count_duplicate_rows(df)
        highest_null=cleaner.find_column_with_most_nulls(df)
        context['highest_name']=highest_null[0]
        context['highest_count']=highest_null[1]
        # Render the template
        return TemplateResponse(request, self.template_name, context)
     
class CleanView(generic.View, LoginRequiredMixin):
    template_name= "home/fetch.html"
    
    def post(self,  request, *args, **kwargs):
        # Call the parent class's get method to set the object attribute
        #response = super().post(request, *args, **kwargs)
        tasks=[]
        # Retrieve the data frame from the session
        df_json = request.session.get('data_frame')
        
        # Deserialize the data frame from JSON
        df = pd.read_json(df_json)
        # Get the value of the "remove-duplicates" checkbox
        remove_duplicates = request.POST.get('remove-duplicates')
        if remove_duplicates:
            tasks.append(remove_duplicates.s())
        # Get the value of the "remove-column" checkbox
        remove_column = request.POST.get('remove-column')
        # If the "remove-column" checkbox is checked, get the value of the "remove-column-input1" select element
        if remove_column:
            remove_column_names = request.POST.getlist('remove-column-input1') 
            tasks.append(remove_columns.s(columns=remove_column_names))

        # Get the value of the "missing-values" checkbox
        handle_missing_values = request.POST.get('missing-values')
        # If the "missing-values-remove" checkbox is checked, get the value of the "missing-values-remove-input1" select element
        if handle_missing_values:
            missing_values_remove = request.POST.get('missing-values-remove')
            if missing_values_remove:
                missing_values_remove_columns = request.POST.getlist('missing-values-remove-input1') 
                tasks.append(remove_null_on_column.s(columns=missing_values_remove_columns))
            
            # If the "missing-values-replace" checkbox is checked, get the value of the "missing-values-replace-input1" select element and the "missing-values-replace-input2" text input
            missing_values_replace = request.POST.get('missing-values-replace')
            if missing_values_replace:
                missing_values_replace_columns = request.POST.getlist('missing-values-replace-input1')
                missing_values_replace_values = request.POST.getlist('missing-values-replace-input2') 
                column_value_pairs=list(zip(missing_values_replace_columns, missing_values_replace_values))
                tasks.append(replace_null_on_columns.s( value_pairs_list=column_value_pairs))
        convert_types= request.POST.get('convert-types')
        if convert_types:
            # The convert-types checkbox is checked
            data_type_columns_and_types = []
            for i in range(200):  # Loop through a large number of form elements
                data_type_column = request.POST.get('dataTypeSelect1' + str(i+1))
                if data_type_column:
                    # The dataTypeSelect1 form element exists
                    data_type = request.POST.get('dataTypeSelect2' + str(i+1))
                    data_type_columns_and_types.append((data_type_column, data_type))
                else:
                    # No more form elements were found, exit the loop
                    break

            tasks.append(convert_data_type.s(data_type_columns_and_types))
        
        # Execute the clean_data_frame task asynchronously
        result = clean_data.delay(df_json, tasks)

        # Wait for the task to complete and get the result
        df = result.get()
        request.session['data_frame2'] = df.to_json()
       
        return TemplateResponse(request, 'home/task_progress.html')
    


def check_task_progress(request,taskIdPrefix ):
    redis_client = redis.Redis(host='localhost', port=6379, db=0)

    # Retrieve the task id from Redis cache
    task_id = redis_client.get(f"{taskIdPrefix}")
    task_id = task_id.decode()
    # Check the status of the task
    result = AsyncResult(task_id)
    status = result.status
    progress = result.info.get('progress', 0) if status == 'PROGRESS' else 100
    return JsonResponse({'status': status, 'progress': progress})


class AFterCleanView(generic.TemplateView):
    template_name= 'home/after_clean.html'
    
    def get(self,  request, *args, **kwargs):
        df_json = request.session.get('data_frame2')
        
        # Deserialize the data frame from JSON
        df = pd.read_json(df_json)
        # Use get_context_data to create a context dictionary
        context = self.get_context_data(**kwargs)

        context['class_attrs'] = {'class': 'special'}
        # Do some processing on the dataframe
        #context['dimensions']= cleaner.calculate_dimensions(df),
        dimensions= cleaner.calculate_dimensions(df)
        context['rows']= dimensions[0]
        context['columns']= dimensions[1]
        context['total_nulls']=cleaner.calculate_null_values_total(df)
        context['null_percentage']= cleaner.calculate_null_percentage(df)
        context['preview']=df.head()
        context['size']=cleaner.dataframe_size(df)
        context['duplicates']=cleaner.count_duplicate_rows(df)
        highest_null=cleaner.find_column_with_most_nulls(df)
        context['highest_name']=highest_null[0]
        context['highest_count']=highest_null[1]
        # Render the template
        return TemplateResponse(request, self.template_name, context)
    
class DataQualityView(generic.TemplateView):
    template_name= 'home/data_quality.html'
    
class VisualizationView(generic.TemplateView):
    template_name= 'home/visualization.html'
    
class AdvancedCleanView(generic.TemplateView):
    template_name= "home/post_clean.html"    
    
class CollaborationView(generic.TemplateView):
    template_name= 'home/collaboration.html'

class IntegrationView(generic.TemplateView):
    template_name= 'home/integration.html' 

class SupportView(generic.TemplateView):
    template_name= 'home/support.html' 