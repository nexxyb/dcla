# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""


from urllib import request
from django import template, forms
from django.http import HttpResponse, HttpResponseServerError , HttpResponseBadRequest
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
        
        # Store the data frame in the session
        request.session['data_frame'] = df.to_json()

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
     
class CleanView(generic.TemplateView):
    template_name= "home/fetch.html"
    
    def post(self,  request, *args, **kwargs):
        # Call the parent class's get method to set the object attribute
        response = super().post(request, *args, **kwargs)
        
        # Retrieve the data frame from the session
        df_json = request.session.get('data_frame')
        
        # Deserialize the data frame from JSON
        df = pd.read_json(df_json)
        
        selected_activities = []
        if 'duplicate_removal' in request.POST:
            selected_activities.append('duplicate_removal')
        if 'missing_value_handling' in request.POST:
            selected_activities.append('missing_value_handling')
        # ...
        
        # Process the selected cleaning activities
        # ...
        request.session['data_frame2'] = df.to_json()
        context = self.get_context_data(object=obj, **kwargs)
        
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