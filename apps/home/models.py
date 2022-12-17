# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from operator import mod
from time import timezone
from unicodedata import decimal
from django.db import models
from django.db.models import Sum, Q
from datetime import date
import datetime
from django.urls import reverse
import uuid
from django.contrib.auth import get_user_model
from django.conf import settings   
#from djmoney.models.fields import MoneyField
from random import randint
from django.template.defaultfilters import slugify
#from moneyed import Money

User =get_user_model()
class File(models.Model):
    name=models.CharField(max_length=100, unique=True)
    size=models.IntegerField()
    created_date=models.DateTimeField(auto_now_add=datetime.cow())
    content_type=models.CharField(max_length=6)
    upload_by=models.ForeignKey(User, on_delete= models.CASCADE)
    s3_key=models.CharField(max_length=100)

    
    def __str__(self):
        return self.name
    
