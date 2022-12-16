# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib.auth.models import AbstractUser, AbstractBaseUser, PermissionsMixin
from django.db.models import CharField
from django.urls import reverse
from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.core.mail import send_mail
from random import randint
from django.template.defaultfilters import slugify

from .managers import CustomUserManager



class User(AbstractBaseUser, PermissionsMixin):
    """
    Default custom user model for fims3.
    If adding fields that need to be filled at user signup,
    check forms.SignupForm and forms.SocialSignupForms accordingly.
    """

    email = models.EmailField(_('email address'), unique=True)
    first_name = models.CharField(_('first name'), max_length=30, blank=True)
    last_name = models.CharField(_('last name'), max_length=30, blank=True)
    date_joined = models.DateTimeField(_('date joined'), auto_now_add=True)
    bio = models.TextField(blank=True, null=True, max_length= 150)
    location= models.CharField(null=True, blank=True, max_length= 20 )
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_pm = models.BooleanField(default=False)
    is_apm = models.BooleanField(default=True)
    #company_name = models.CharField(verbose_name= 'Company Name', blank=True, null=True, max_length= 45 )
    #company_code = models.IntegerField(verbose_name='Company Code', blank=True, default=randint(10102, 90909), unique=True)
    slug = models.SlugField(null=False, unique=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()


    def get_absolute_url(self):
        """Get url for user's detail view.

        Returns:
            str: URL for user detail.

        """
        return reverse("profile", kwargs={'slug':self.slug})
    
    
    def get_full_name(self):
        '''
        Returns the first_name plus the last_name, with a space in between.
        '''
        full_name = f'{self.first_name} {self.last_name}'
        return full_name.strip()

    def get_short_name(self):
        '''
        Returns the short name for the user.
        '''
        return self.first_name

    def email_user(self, subject, message, from_email=None, **kwargs):
        '''
        Sends an email to this User.
        '''
        send_mail(subject, message, from_email, [self.email], **kwargs)

    def save(self, *args, **kwargs):  # new
        if not self.slug:
            self.slug = slugify(self.email)
        return super().save(*args, **kwargs)