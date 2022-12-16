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

