from flask import Flask, request, redirect, render_template, flash

from signup import signup_user

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Call the signup_user function (make sure this function is defined/imported)
        signup_user(username, email, password)

        # Redirect to a success page
        return redirect('/signup_success.html')
    
    # Render the signup page
    return render_template('signup.html')

if __name__ == '__main__':
    app.run(debug=True)
