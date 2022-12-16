# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path
from .views import login_view, Profile, UpdateProfileView,register_user, AccountUpgradeView
from django.contrib.auth.views import LogoutView
from django.contrib.auth import views

urlpatterns = [
    path('accounts/login/', login_view, name="login"),
    path('accounts/register/', register_user, name="register"),
    #path('accounts/register/', SignUpView.as_view(), name="register"),
    path("accounts/logout/", LogoutView.as_view(), name="logout"),
    path('profile/<slug:slug>/', Profile.as_view(), name = 'profile'),
    path('profile/<slug:slug>/edit', UpdateProfileView.as_view(), name= 'edit-profile'),
    path('account/<slug:slug>/upgrade/', AccountUpgradeView.as_view(), name='account-upgrade'),
]

