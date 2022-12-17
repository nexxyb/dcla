# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""


from urllib import request
from django import template, forms
from django.http import HttpResponse, HttpResponseServerError , HttpResponseBadRequest
from django.urls import reverse, reverse_lazy
from django.views import generic
from django.contrib.auth.mixins import LoginRequiredMixin
#from .models import Expense, Income, Project
from django.views.generic.edit import CreateView, DeleteView, UpdateView
from django.contrib.auth import get_user_model
from django.shortcuts import redirect, render
from .forms import ContactForm
from django.core.mail import send_mail, BadHeaderError
import boto3
from datetime import timezone
from django.dispatch import receiver
from django.db.models.signals import post_save
from django.conf import settings
import pandas as pd
from .models import File
import tempfile
from . import cleaner


User=get_user_model()
s3_client = boto3.resource('s3')

class IndexView(generic.TemplateView):
    template_name='home/raw2.html'
    
class FileUploadView(generic.TemplateView):
    
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
        
        if uploaded_file.content_type not in ['application/vnd.ms-excel', 'text/csv']:
            # File is not a valid image
            return HttpResponseBadRequest('Invalid file type')
        
        # Rename the file using the current timestamp
        file_name = str(timezone.now().timestamp())
        key = settings.AWS_FILE_UPLOAD_LOCATION + file_name
        # Use the AWS SDK for Python (Boto3) to upload the file to the S3 bucket
        try:
            s3_client.meta.client.upload_file(uploaded_file, settings.AWS_STORAGE_BUCKET_NAME, key)
        except Exception as e:
            # Failed to upload the file to S3
            return HttpResponseBadRequest('Failed to upload file to S3')
        
        # Save the uploaded file details to the database
        file = File.objects.create(
                                    name=file_name,
                                    size=uploaded_file.size,
                                    content_type=uploaded_file.content_type,
                                    upload_by=request.user,
                                    s3_key=key
                                )
        
        return HttpResponse('File uploaded successfully')


class FetchFileView(generic.TemplateView):
    template_name='home/raw2.html'
    
    def get(self, request,key):
        # Connect to AWS S3
        #s3 = boto3.client('s3')

        # Get the file key from the request
        #file_key = request.GET.get('file_key')

        # Check if the file key was provided
        if not key:
            return HttpResponseBadRequest("Missing file key")

        try:
            # Download the file from S3
            s3_client.meta.client.download_file(settings.AWS_STORAGE_BUCKET_NAME, key, '/tmp/temp_file.csv')
        except Exception as e:
            # Return a 500 error if the file cannot be downloaded
            return HttpResponseServerError("Error downloading file: {}".format(e))

        # Load the file into a Pandas dataframe
        df = pd.read_csv('/tmp/temp_file.csv')
        
        # Store the data frame in the session
        request.session['data_frame'] = df.to_json()

        # Do some processing on the dataframe
        context = {
                'dimensions': cleaner.calculate_dimensions(df),
                'total_nulls':cleaner.calculate_null_values_total(df),
                'null_percentage':cleaner.calculate_null_percentage(df),
                'preview':df.head(),
        }
        
        # Render the template
        return render(request, self.template_name context=context)
    
    def  get_context_data(self, **kwargs):
         # Call the base implementation first to get a context
        context=super(FetchFileView , self).get_context_data(**kwargs)
        # Add in a QuerySet of all other contexts
        context['object']= 'Classroom'
        return context
    
@receiver(post_save, sender=File)
def filefetch_view(sender, instance, created, **kwargs):
    if created:
        filefetch_view = FetchFileView()
        filefetch_view.get(request, key=instance.s3_key)   
class CleanView(generic.TemplateView):
    template_name= "home/clean.html"
    
    def get(self, request):
        # Retrieve the data frame from the session
        df_json = request.session.get('data_frame')

        # Deserialize the data frame from JSON
        df = pd.read_json(df_json)
        
        # Do some processing on the data frame

        # Render the template
        return render(request, self.template_name)

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