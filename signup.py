import sqlite3
import bcrypt

def signup_user(username, email, password):
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Connect to the database
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    try:
        # Insert the new user into the database
        cursor.execute('''
        INSERT INTO users (username, email, password) VALUES (?, ?, ?)
        ''', (username, email, hashed_password))

        # Commit changes
        conn.commit()
        print("User signed up successfully.")

    except sqlite3.IntegrityError:
        print("Username or email already exists.")

    finally:
        # Close the connection
        conn.close()

# Testing the signup_user function
if __name__ == "__main__":
    # Replace these with user input or web form data in a real application
    username = input("Enter username: ")
    email = input("Enter email: ")
    password = input("Enter password: ")

    signup_user(username, email, password)
