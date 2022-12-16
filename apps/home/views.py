# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""


from urllib import request
from django import template, forms
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest
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

s3_client = boto3.client('s3')

class IndexView(generic.TemplateView):
    template_name='home/raw2.html'
    
class FileUploadView(generic.TemplateView):
    pass 
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
        
        # Use the AWS SDK for Python (Boto3) to upload the file to the S3 bucket
        try:
            s3_client.upload_fileobj(uploaded_file, 'my-bucket', file_name)
        except Exception as e:
            # Failed to upload the file to S3
            return HttpResponseBadRequest('Failed to upload file to S3')
        
        # Save the uploaded file details to the database
        file = File(
            name=file_name,
            size=uploaded_file.size,
            content_type=uploaded_file.content_type
        )
        file.save()
        
        return HttpResponse('File uploaded successfully')


class FetchFileView(generic.TemplateView):
    pass 
    
class CleanView(generic.TemplateView):
    template_name= "home/clean.html"
    
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