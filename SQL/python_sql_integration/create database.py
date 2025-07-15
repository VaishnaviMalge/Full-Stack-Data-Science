# create new database 
import mysql.connector

conn = mysql.connector.connect(host = 'localhost', user='root', password='9960')

mycursor.execute('create database pythondb')
print(mycursor)
