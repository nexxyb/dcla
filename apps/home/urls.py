# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.home import views

urlpatterns = [

    # The home page
    path('', views.IndexView.as_view(), name='home'),
    path('clean', views.CleanView.as_view(), name='clean'),
    path('advanced-clean', views.AdvancedCleanView.as_view(), name='advanced-clean'),
    path('file-upload', views.FileUploadView.as_view(), name='file-upload'),
    path('file-fetch/<int:pk>/', views.FetchFileView.as_view(), name='file-fetch'),
    path('data-quality', views.DataQualityView.as_view(), name='data-quality'),
    path('visualization', views.VisualizationView.as_view(), name='visualization'),
    path('integration', views.IntegrationView.as_view(), name='integration'),
    path('collaboration', views.CollaborationView.as_view(), name='collaboration'),
    path('support', views.SupportView.as_view(), name='support'),
    path('task-progress/<str:task_id>', views.check_task_progress, name='task-progress')
]
