# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.apps import AppConfig


class MyConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.home'
    label = 'apps_home'

    def ready(self):
        # Connect the signal to the receiver function
        from . import signals