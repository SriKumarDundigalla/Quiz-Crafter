import sqlite3
import csv
from werkzeug.security import generate_password_hash
import random
import string

DATABASE = 'users.db'
def get_db():
    conn = sqlite3.connect(DATABASE)
    return conn
def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database on app start
init_db()

# Function to insert a user into the database
def insert_user(username, password):
    # Use 'pbkdf2:sha256' instead of 'sha256'
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

# Function to generate a random 6-character password
def generate_password(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Create 10 users and insert them into the database
users = []
for i in range(1, 11):
    username = f'User{i}'
    password = generate_password()
    users.append((username, password))
    insert_user(username, password)

# Write the users to a CSV file
with open('users.csv', 'w', newline='') as csvfile:
    fieldnames = ['username', 'password']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for user in users:
        writer.writerow({'username': user[0], 'password': user[1]})

print("Users inserted into database and CSV generated as 'users.csv'.")
