# db_startup.py
import sqlite3

def create_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Drop the attendance and announcements tables if they exist
    c.execute('''DROP TABLE IF EXISTS attendance''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (name TEXT, date TEXT, time TEXT, subject TEXT, late BOOLEAN DEFAULT 0, UNIQUE(name, date, subject))''')

    c.execute('''DROP TABLE IF EXISTS announcements''')
    c.execute('''CREATE TABLE IF NOT EXISTS announcements
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  date_time TEXT, 
                  teacher_name TEXT, 
                  message TEXT)''')

    # Create the users table for authentication
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  password TEXT, 
                  email TEXT UNIQUE)''')

    conn.commit()
    conn.close()

# Optional: You can also make this script runnable as a standalone script
if __name__ == '__main__':
    create_db()
