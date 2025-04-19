Please follow these instructions. In case you have any issue, you can ask an AI assistant to help you. 
  1. Check Python Installation
Open the Terminal application. You can find it in Applications/Utilities or by searching for "Terminal" using Spotlight (Cmd + Space).
Type the following command and press Enter:
Bash

python3 --version
If you see a version number like Python 3.x.x (where x.x is 8 or higher, preferably), you're good to go.
If the command fails or shows Python 2.x.x, you should install the latest version of Python 3.
2. Install Python 3 (If Necessary)

Option A (Recommended): Homebrew (If you don't have Homebrew, install it from https://brew.sh/) In Terminal, run:

Bash

brew install python
Homebrew usually handles setting up your PATH correctly. Close and reopen the Terminal after installation.

Option B: Official Installer

Go to the official Python website: https://www.python.org/downloads/macos/
Download the latest macOS installer package (.pkg).
Run the installer, following the default steps. The installer should handle adding Python 3 to your PATH.
After installing: Close your current Terminal window and open a new one. Run python3 --version again to confirm the installation worked.

3. Set Up Your Project Folder

Create a dedicated folder for your project, for example, in your Documents folder.
Bash

# In Terminal, navigate to where you want to create the folder (e.g., Documents)
cd ~/Documents

# Create the folder
mkdir CoursePlannerApp

# Navigate into the new folder
cd CoursePlannerApp
Place your files inside this folder:
app.py
cs_courses.csv
general_courses.csv
4. Create and Activate a Virtual Environment (Highly Recommended)

This keeps project dependencies isolated.

Make sure you are inside your project directory (CoursePlannerApp) in the Terminal.
Create the virtual environment (let's call it venv):
Bash

python3 -m venv venv
Activate the virtual environment:
Bash

source venv/bin/activate
You'll know it's active because (venv) will appear at the beginning of your Terminal prompt line.
5. Install Required Libraries

While the virtual environment is active, install the necessary Python packages using pip.

Run this command in the same Terminal window:
Bash

pip install streamlit pandas numpy scikit-learn nltk
(If pip gives an error or points to an old version, try pip3 install ... instead)
This installs Streamlit and the other libraries (pandas, numpy, scikit-learn, nltk).
6. Run Your Streamlit App

Ensure you are still in your project directory (CoursePlannerApp) in Terminal and the virtual environment (venv) is active.
Run the application:
Bash

streamlit run app.py
Streamlit will start the server, and you'll see output like:
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://<your-local-ip>:8501
Your default web browser should open automatically to the Local URL (http://localhost:8501).
The app might pause briefly on the first run to download NLTK data ('stopwords').
7. Using and Stopping the App

Use the app in your browser.
To stop the Streamlit server, go back to the Terminal window where it's running and press Ctrl + C.
8. Deactivating the Virtual Environment (Optional)

When you're done, you can deactivate the virtual environment by simply typing in the Terminal:
Bash

deactivate
The (venv) prefix will disappear. Remember to reactivate it (source venv/bin/activate) the next time you want to work on this project.
You should now have your course planner running locally on your Mac!
