
# Deep Learning In Python - Week 2

<img src="https://github.com/hannesrosenbusch/DLIP_Week1/blob/main/assets/uvalogo.svg.png?raw=true" width="125">    <img src="https://github.com/hannesrosenbusch/DLIP_Week1/blob/main/assets/pythonlogo.jpeg?raw=true" width="380">


An introduction to Python can be seen here: <br>
[Python for R users](https://youtube.com/playlist?list=PLq0cz82QvYapppmpXPYgS76VbHHKRIgbk)

# Steps for Completing the Assignment

## Step 1: Create your own repo
On this repository's GitHub page, click **Use this template** and create a repository in your own account (you can rename it if convenient; it should be a PUBLIC repo).

Work in your own copy, there is no need to ever return to the course repo. You will clone, commit, and push to the repo in your own account.

## Step 2: Enable GitHub Actions
Open your new repository's **Actions** tab. If GitHub asks whether to enable workflows, confirm.

If workflows are disabled in repository settings, allow GitHub-provided actions (including `actions/checkout` and `actions/setup-python`), or select **Allow all actions and reusable workflows**. Save the setting. Then return to **Actions** and enable the workflow if prompted.

## Step 3: Get a working copy onto your computer
Clone your new repository using its GitHub page's **Code** button, or edit it in an online environment such as Google Colab or Kaggle. Replit is another option if your computer does not work well. If you work locally, install Python first. See this guide for basic Git commands: https://www.earthdatascience.org/workshops/intro-version-control-git/basic-git-commands/.

## Step 4: Complete and test the assignment
Edit (only!) the tasks.py file by filling in your solution code.
Use the functions, and print the returned outputs, to see whether you have errors.
You can also run all grading tests locally with these commands:
 ```bash
 python -m pip install numpy pytest
 python -m pytest tests/grade_task1.py tests/grade_task2.py tests/grade_task3.py
 ```
The grading files are in the `tests/` folder; don't edit them.
GitHub Actions runs all three grading files automatically on pushes and pull requests.
The Actions run summary reports the outcome of each task.

## Step 5: Push your work and check your grade
Commit and push your changes to your own repository. For a local clone, run:
 ```bash
 git add tasks.py
 git commit -m "Completed tasks"
 git push
 ```
After the push, go to your browser, open your repository's **Actions** tab and select the newest **Grade assignment** run. Its **Summary** shows the result for each task; open the task steps to see test details. If no run appears, check that Actions are enabled and that you pushed to your own repository.

## Step 6: Upload to canvas
Follow the instructions on canvas (i.e., likely you will have to upload a link to this repo. Make sure the repo settings say it's public rather than private).
