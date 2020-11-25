"""
db module

Description
===========
This module stores the database authentication

Made by:
    Muhammad Nabillah Fihira Rischa
    abel.rischa@gmail.com
"""
import mysql.connector as connector

connection = connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="db_cspc"
)
