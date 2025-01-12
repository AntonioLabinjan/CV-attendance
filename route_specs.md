### Specification for `/set_subject` Route

- **Supported Methods**: `GET`, `POST`
- **Functions Used**:
  - `request.form`: Retrieves form data (`subject`, `date`, `start_time`, `end_time`) from the POST request.
  - `render_template()`: Renders the HTML templates `set_subject.html` or `set_subject_success.html`.
- **Templates Rendered**:
  - `set_subject.html`: Rendered when the route is accessed via a `GET` request, presenting the form to the user.
  - `set_subject_success.html`: Rendered upon a successful `POST` request, confirming the subject and timings have been set.
- **Route Usage**: This route allows users to set the current subject for logging, including the date and time, and displays a success page upon submission.

---

### Specification for `/add_student` Route

- **Supported Methods**: `GET`, `POST`
- **Functions Used**:
  - `request.form`: Retrieves the student's name from the form data.
  - `request.files.getlist('images')`: Retrieves the list of images uploaded by the user.
  - `os.path.join()`: Constructs file paths for saving images and creating directories.
  - `os.makedirs()`: Creates a new directory for the student within the `known_faces` folder, if it doesn't already exist.
  - `image.save(image_path)`: Saves each uploaded image to the designated folder.
  - `add_known_face(image_path, name)`: Processes the saved images for face recognition and associates them with the student's name.
  - `render_template()`: Renders the HTML templates `add_student.html` or `add_student_success.html`.
- **Templates Rendered**:
  - `add_student.html`: Rendered when the route is accessed via a `GET` request, presenting the form to add a new student.
  - `add_student_success.html`: Rendered upon a successful `POST` request, confirming the student has been added and their images have been saved.
- **Route Usage**: This route allows users to add a new student to the database, save their images for face recognition in a designated folder, and display a success page upon completion.

---

### Specification for `/add_student_success` Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `render_template()`: Renders the HTML template `add_student_success.html`.
- **Templates Rendered**: 
  - `add_student_success.html`: Displayed to confirm the successful addition of a student.
- **Route Usage**: This route renders a success page confirming that a new student has been successfully added to the database.

---

### Specification for `/video_feed` Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `generate_frames()`: A function that captures video frames, processes them for face recognition, and adds bounding boxes and labels to detected faces.
  - `Response()`: Streams the video frames with the specified `mimetype` to be displayed in real-time on the client side.
- **Templates Rendered**: None (this route directly streams video data).
- **Route Usage**: This route streams video frames captured from the camera, with face recognition applied, displaying the frames with bounding boxes and labels for recognized faces.

---

### Specification for `/` (Main) Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `render_template()`: Renders the HTML template `index.html`.
- **Templates Rendered**:
  - `index.html`: The main homepage of the application.
- **Route Usage**: This is the main route that renders the homepage of the application when accessed.

---

### Specification for `/attendance` Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `sqlite3.connect()`: Connects to the SQLite database `attendance.db`.
  - `c.execute()`: Executes a SQL query to retrieve attendance records from the database, including the subject, student name, date, time, and whether the student was late.
  - `c.fetchall()`: Fetches all the resulting records from the executed query.
  - `conn.close()`: Closes the database connection after fetching the records.
  - `render_template()`: Renders the `attendance.html` template with the grouped attendance data.
- **Templates Rendered**:
  - `attendance.html`: Displays the attendance records grouped by date and subject.
- **Route Usage**: This route retrieves all attendance records from the database, organizes them by date and subject, and renders them in a structured format for easy viewing and analysis.

---

### Specification for `/download` Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `sqlite3.connect()`: Connects to the SQLite database `attendance.db`.
  - `c.execute()`: Executes a SQL query to retrieve attendance records, sorted by subject, date, and time.
  - `c.fetchall()`: Fetches all the resulting records from the executed query.
  - `conn.close()`: Closes the database connection after fetching the records.
  - `io.StringIO()`: Creates an in-memory string buffer to hold the CSV data.
  - `csv.writer()`: Writes data to the CSV format in the string buffer.
  - `writer.writerow()`: Writes individual rows of data to the CSV file, including headers, subject sections, and attendance records.
  - `output.seek(0)`: Resets the buffer’s position to the beginning before sending the response.
  - `Response()`: Sends the generated CSV file as a downloadable response with the appropriate MIME type (`text/csv`) and headers (`Content-Disposition`).
- **Templates Rendered**: None (this route directly generates and sends a CSV file).
- **Route Usage**: This route fetches all attendance records from the database, organizes them by subject, and exports them as a CSV file that users can download.

---

### Specification for `/delete_attendance/<int:id>` Route

- **Supported Methods**: `POST`
- **Functions Used**:
  - `sqlite3.connect()`: Connects to the SQLite database `attendance.db`.
  - `c.execute()`: Executes a SQL `DELETE` command to remove the attendance record that matches the provided `rowid`.
  - `conn.commit()`: Commits the changes to the database, making the deletion permanent.
  - `conn.close()`: Closes the database connection after the deletion.
  - `redirect()`: Redirects the user to the `/attendance` page after the deletion is completed.
  - `url_for()`: Generates the URL for the `/attendance` route.
- **Templates Rendered**: None (this route performs a redirection after deletion).
- **Route Usage**: This route handles the deletion of a specific attendance record from the database using the record's `rowid`, and then redirects the user back to the attendance overview page to reflect the update.

---

### Specification for `/statistics` Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `sqlite3.connect()`: Connects to the SQLite database `attendance.db`.
  - `c.execute()`: Executes SQL queries to retrieve attendance statistics:
    - The first query counts the number of attendance records per student for each subject.
    - The second query counts the total number of attendance records for each subject.
  - `c.fetchall()`: Fetches all the resulting records from the executed queries.
  - `conn.close()`: Closes the database connection after retrieving the data.
  - `render_template()`: Renders the `statistics.html` template with the retrieved attendance statistics data.
- **Templates Rendered**:
  - `statistics.html`: Displays the attendance statistics, including the count of records per student for each subject and the total count per subject.
- **Route Usage**: This route retrieves and displays attendance statistics from the database, showing the number of records grouped by student and subject, as well as overall subject attendance, presented in a structured format on the statistics page.

---

### Specification for `/students` Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `sqlite3.connect()`: Connects to the SQLite database `attendance.db`.
  - `c.execute()`: Executes a SQL query to select distinct student names from the `attendance` table, ordered alphabetically.
  - `c.fetchall()`: Fetches all unique student names from the query result.
  - `conn.close()`: Closes the database connection after retrieving the data.
  - `render_template()`: Renders the `students.html` template, passing the list of student names.
- **Templates Rendered**:
  - `students.html`: Displays the sorted list of unique student names.
- **Route Usage**: This route retrieves a list of unique student names from the database, sorted alphabetically, and displays them on the `students.html` page.

---


### Specification for `/plot/student_attendance` Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `sqlite3.connect()`: Connects to the SQLite database `attendance.db`.
  - `c.execute()`: Executes a SQL query to retrieve the count of attendances per student from the database.
  - `c.fetchall()`: Fetches all the resulting records from the executed query.
  - `conn.close()`: Closes the database connection after fetching the records.
  - `plt.figure()`: Creates a new figure for the plot.
  - `sns.barplot()`: Creates a bar plot using Seaborn, displaying the count of attendances for each student.
  - `plt.title()`, `plt.xlabel()`, `plt.ylabel()`: Set the title and axis labels for the plot.
  - `io.BytesIO()`: Creates an in-memory byte stream to hold the plot image.
  - `plt.savefig()`: Saves the plot image to the byte stream.
  - `img.seek(0)`: Resets the byte stream’s position to the beginning before sending the response.
  - `plt.close()`: Closes the plot to free up resources.
  - `send_file()`: Sends the plot image as a response with MIME type `image/png`.
- **Templates Rendered**: None (this route directly returns a plot image).
- **Route Usage**: This route generates and serves a bar plot showing the number of attendances per student.

---

### Specification for `/plot/subject_attendance` Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `sqlite3.connect()`: Connects to the SQLite database `attendance.db`.
  - `c.execute()`: Executes a SQL query to retrieve the count of attendances per subject from the database.
  - `c.fetchall()`: Fetches all the resulting records from the executed query.
  - `conn.close()`: Closes the database connection after fetching the records.
  - `plt.figure()`: Creates a new figure for the plot.
  - `plt.pie()`: Creates a pie chart using Matplotlib, displaying the proportion of attendances for each subject.
  - `plt.title()`: Sets the title of the pie chart.
  - `io.BytesIO()`: Creates an in-memory byte stream to hold the plot image.
  - `plt.savefig()`: Saves the plot image to the byte stream.
  - `img.seek(0)`: Resets the byte stream’s position to the beginning before sending the response.
  - `plt.close()`: Closes the plot to free up resources.
  - `send_file()`: Sends the plot image as a response with MIME type `image/png`.
- **Templates Rendered**: None (this route directly returns a plot image).
- **Route Usage**: This route generates and serves a pie chart showing the distribution of attendances across different subjects.

---

### Specification for `/plot/monthly_attendance` Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `sqlite3.connect()`: Connects to the SQLite database `attendance.db`.
  - `c.execute()`: Executes a SQL query to retrieve the count of attendances per month from the database.
  - `c.fetchall()`: Fetches all the resulting records from the executed query.
  - `conn.close()`: Closes the database connection after fetching the records.
  - `plt.figure()`: Creates a new figure for the plot.
  - `plt.pie()`: Creates a pie chart using Matplotlib, displaying the proportion of attendances for each month.
  - `plt.title()`: Sets the title of the pie chart.
  - `io.BytesIO()`: Creates an in-memory byte stream to hold the plot image.
  - `plt.savefig()`: Saves the plot image to the byte stream.
  - `img.seek(0)`: Resets the byte stream’s position to the beginning before sending the response.
  - `plt.close()`: Closes the plot to free up resources.
  - `send_file()`: Sends the plot image as a response with MIME type `image/png`.
- **Templates Rendered**: None (this route directly returns a plot image).
- **Route Usage**: This route generates and serves a pie chart showing the distribution of attendances across different months.

---

### Specification for `/plots` Route

- **Supported Methods**: `GET`
- **Functions Used**:
  - `render_template()`: Renders the HTML template `plot_router.html`.
- **Templates Rendered**:
  - `plot_router.html`: Provides links to the different plot routes available.
- **Route Usage**: This route renders a page with links to the various plot routes, allowing users to view different visualizations.

---

### Specification for `/predict_absence` Route
**Supported Methods:** GET  
**Functions Used:**
- **get_weather_forecast(api_key, location="your_city")**: Calls an external weather API to get the weather forecast for a specific location.
- **predict_absence_due_to_weather(weather_condition)**: Determines if the weather condition is likely to cause absences based on predefined bad weather keywords.
- **requests.get()**: Sends a GET request to the weather API.
- **jsonify()**: Converts the response data to a JSON format for easy consumption by clients.

**Templates Rendered:** None  
**Route Usage:** This route fetches the current weather forecast using a weather API and predicts if there will be possible late arrivals or absences due to bad weather. It returns a JSON response containing the weather condition and a message advising on potential attendance issues.

---

### Specification for `/announcements` Route
**Supported Methods:** GET  
**Functions Used:**
- **sqlite3.connect()**: Connects to the SQLite database `attendance.db`.
- **c.execute()**: Executes a SQL query to select all announcements ordered by date and time in ascending order.
- **c.fetchall()**: Fetches all rows from the announcements table as a list of tuples.
- **conn.close()**: Closes the database connection after retrieving the data.
- **render_template()**: Renders the `announcements.html` template, passing the list of announcements.

**Templates Rendered:**
- **announcements.html**: Displays all announcements in chronological order.

**Route Usage:** This route retrieves all announcements from the database, ordered by date and time, and displays them on the `announcements.html` page.

---

### Specification for `/post_announcement` Route
**Supported Methods:** POST  
**Functions Used:**
- **request.form['message']**: Retrieves the message content from the submitted form data.
- **datetime.now().strftime("%Y-%m-%d %H:%M:%S")**: Gets the current date and time, formatted as a string.
- **current_user.username**: Retrieves the username of the currently logged-in user.
- **sqlite3.connect()**: Connects to the SQLite database `attendance.db`.
- **c.execute()**: Executes a SQL query to insert a new announcement into the announcements table.
- **conn.commit()**: Commits the transaction to save the new announcement.
- **conn.close()**: Closes the database connection after saving the data.
- **redirect(url_for('announcements'))**: Redirects the user back to the announcements page after posting.

**Templates Rendered:** None  
**Route Usage:** This route allows authenticated users to post a new announcement. The announcement is saved to the database with the current date, time, and the teacher's name, then the user is redirected to the `announcements` page to see the updated list.

---

### Specification for `/delete_announcement/<int:id>` Route
**Supported Methods:** POST  
**Functions Used:**
- **sqlite3.connect()**: Connects to the SQLite database `attendance.db`.
- **c.execute()**: Executes a SQL query to delete an announcement by its ID.
- **conn.commit()**: Commits the transaction to remove the announcement from the database.
- **conn.close()**: Closes the database connection after deleting the data.
- **redirect(url_for('announcements'))**: Redirects the user back to the announcements page after deletion.

**Templates Rendered:** None  
**Route Usage:** This route allows authenticated users to delete an announcement by its ID. After deletion, the user is redirected to the `announcements` page.

---

### Specification for `/edit_announcement/<int:id>` Route
**Supported Methods:** GET, POST  
**Functions Used:**
- **sqlite3.connect()**: Connects to the SQLite database `attendance.db`.
- **request.method == 'POST'**: Checks if the request method is POST, indicating that the form was submitted.
- **request.form['message']**: Retrieves the updated message content from the submitted form data.
- **datetime.now().strftime("%Y-%m-%d %H:%M:%S")**: Gets the current date and time, formatted as a string.
- **current_user.username**: Retrieves the username of the currently logged-in user.
- **c.execute()**: Executes a SQL query to update the announcement if the user is the owner.
- **conn.commit()**: Commits the transaction to save the updated announcement.
- **conn.close()**: Closes the database connection after saving the data.
- **redirect(url_for('announcements'))**: Redirects the user back to the announcements page after editing.
- **render_template('edit_announcement.html', announcement=announcement)**: Renders the `edit_announcement.html` template, passing the current announcement data for editing.

**Templates Rendered:**
- **edit_announcement.html**: Displays the form for editing the announcement's message.

**Route Usage:** This route allows authenticated users to edit an announcement they posted. It first checks if the user owns the announcement, then updates it. If the method is GET, the current announcement is displayed in an edit form. If POST, the changes are saved and the user is redirected to the `announcements` page.

---

### Specification for `/report` Route
**Supported Methods:** GET  
**Functions Used:**
- **sqlite3.connect()**: Connects to the SQLite database `attendance.db`.
- **cur.execute()**: Executes a SQL query to select distinct subjects and count lectures and student attendance.
- **cur.fetchall()**: Fetches all results of the executed queries.
- **conn.close()**: Closes the database connection after retrieving the data.
- **render_template('attendance_report.html', report=report)**: Renders the `attendance_report.html` template, passing the report data.

**Templates Rendered:**
- **attendance_report.html**: Displays the attendance report for each subject, including student attendance percentages and a status indicating if they meet the attendance requirement.

**Route Usage:** This route generates a detailed attendance report by subject. It counts the total lectures for each subject and the number of lectures each student attended, calculates attendance percentages, and checks if the student meets the 50% attendance requirement. The report is displayed on the `attendance_report.html` page.

---

### Specification for `/late_analysis` Route

- **Supported Methods**: `GET`, `POST`
- **Functions Used**:
  - **`sqlite3.connect('attendance.db')`**: Connects to the SQLite database `attendance.db`.
  - **`c.execute()`**: Executes SQL queries to fetch all entries marked as "late" from the `attendance` table.
  - **`c.fetchall()`**: Fetches all rows resulting from the executed query as a list of tuples.
  - **`Counter()`**: Utilizes Python’s `collections.Counter` to count the frequency of late entries by hour and weekday.
  - **`datetime.strptime()`**: Converts string representations of date and time into `datetime` objects for further analysis.
  - **`plt.bar()`**: Creates a bar chart using Matplotlib to visualize the distribution of late entries by hour.
  - **`plt.xticks()`**: Ensures all hours from 0 to 23 are labeled on the x-axis of the plot.
  - **`plt.xlabel()`, `plt.ylabel()`, `plt.title()`**: Sets the labels and title for the plot.
  - **`io.BytesIO()`**: Creates an in-memory binary stream to store the generated plot image.
  - **`plt.savefig()`**: Saves the generated plot as a PNG image in the binary stream.
  - **`base64.b64encode()`**: Encodes the binary image data as a base64 string for easy embedding in HTML.
  - **`render_template()`**: Renders the `late_entries.html` template, passing relevant data for display.

- **Templates Rendered**:
  - **`late_entries.html`**: Displays the list of late entries, the most common hour and weekday for late arrivals, and a bar chart visualizing the frequency of late entries by hour.

- **Route Usage**:
  - **Data Collection**:
    - Retrieves all entries from the `attendance` database where the `late` flag is set to 1.
    - Analyzes the `time` and `date` fields to determine the most common hour and weekday for late arrivals.
  - **Data Visualization**:
    - Creates a bar chart showing the distribution of late entries by hour of the day.
    - Encodes the plot image as a base64 string for embedding directly into the rendered HTML.
  - **Rendering**:
    - Renders the `late_entries.html` template, displaying:
      - The list of late entries.
      - The most common hour and weekday for late arrivals.
      - The bar chart of late entries by hour.
  - **Edge Cases**:
    - If no late entries are found, the route handles the absence of data gracefully, ensuring the page still loads without errors.

---

### Specification for `/download_and_email` Route

**Supported Methods:**  
- GET

**Functions Used:**
- `sqlite3.connect()`: Establishes a connection to the `attendance.db` SQLite database.
- `c.execute()`: Executes a SQL query to retrieve attendance records, ordered by subject, date, and time.
- `c.fetchall()`: Fetches all rows from the executed SQL query.
- `io.StringIO()`: Creates an in-memory string buffer to store CSV data.
- `csv.writer()`: Writes CSV data to the string buffer.
- `request.args.get('email')`: Retrieves the user's email address from the query parameters.
- `render_template()`: Renders the HTML template `download.html` with the generated CSV data and user's email.

**Templates Rendered:**
- `download.html`: Renders a page that allows the user to download the generated CSV file and includes a button to open Gmail with the user's email pre-filled as the sender.

**Route Usage:**
- This route allows users to download attendance data in CSV format and provides an option to open Gmail in a new tab with the user's email pre-filled. The attendance data is grouped by subject and ordered by date and time. The user's email is passed through the query parameters or session, and the generated CSV data is embedded in the HTML template for download.

**Functions Used:**
- **`request.args.get('email')`**: Retrieves the email address passed as a query parameter in the GET request.
- **`EmailMessage()`**: Creates a new email message object.
- **`msg.set_content()`**: Sets the plain text content of the email.
- **`msg.add_attachment()`**: Attaches the generated CSV file to the email.
- **`smtplib.SMTP()`**: Connects to an SMTP server for sending the email.
- **`smtplib.SMTP_SSL()`**: Optionally, connects to the SMTP server using SSL encryption.
- **`smtp.send_message()`**: Sends the email message.
- **`smtp.quit()`**: Closes the connection to the SMTP server.
- **`Response()`**: Sends a JSON response back to the client indicating the success or failure of the email operation.

**Templates Rendered:**  
- None (this route directly generates and sends a CSV file and an email).

**Route Usage:**
- **Data Generation:**
  - Retrieves attendance records from the database and writes them into a CSV format.
  - Stores the CSV data in an in-memory string buffer.
- **Email Preparation:**
  - Prepares an email with the attendance CSV file attached.
  - Uses the email address provided as a query parameter to send the email.
- **Response Handling:**
  - Sends a JSON response back to the client, indicating whether the email was sent successfully or if there was an error.
- **Edge Cases:**
  - Handles cases where no email is provided or if the email sending fails, ensuring appropriate error messages are returned in the JSON response.
  - 
  ### Specification for `/scrape-github-profile` Route

#### Route Overview:
- **Route Name**: `/scrape-github-profile`
- **HTTP Method**: `GET`
- **Function Name**: `scrape_github_profile_route()`
- **Description**: This route is responsible for scraping profile information from a specific GitHub profile page and rendering that data on a predefined HTML template or returning an error message if scraping fails.

---

### Specifications:

1. **Request Method**: `GET`
   - This route accepts HTTP GET requests and does not require any query parameters or payload.

2. **GitHub Profile URL**: 
   - The GitHub profile being scraped is hardcoded as `https://github.com/AntonioLabinjan`.
   - In future extensions, the URL could be passed as a parameter to make the route more dynamic.

3. **Scraping Logic**:
   - The `scrape_github_profile(url)` function is called to scrape the GitHub profile information.
   - The function should return a dictionary with the following keys:
     - `name`: GitHub user's full name.
     - `bio`: GitHub user's bio.
     - `followers`: GitHub user's follower count.
   - If the scraping operation fails (returns `None` or an empty dictionary), a JSON response with an error message will be returned.

4. **Response - Successful Scraping**:
   - If the scraping is successful, the following information is passed to the HTML template (`profile.html`):
     - `name`: The GitHub user's full name.
     - `bio`: The GitHub user's bio.
     - `followers`: The GitHub user's follower count.
   - The HTML template will render the scraped data for display.

5. **Response - Failed Scraping**:
   - If scraping fails, the route will return a JSON response with an error message and a `500` HTTP status code:
     ```json
     {
         "error": "Failed to scrape the data"
     }
     ```

6. **HTML Template (`profile.html`)**:
   - The HTML template will dynamically display the scraped data using the following variables:
     - `{{ name }}`: GitHub user's name.
     - `{{ bio }}`: GitHub user's bio.
     - `{{ followers }}`: Number of followers.

---

### Example Response Flow:

1. **Successful Scraping**:
   - User sends a GET request to `/scrape-github-profile`.
   - The server successfully scrapes the GitHub profile and renders the following in `profile.html`:
     ```html
     <h1>{{ name }}</h1>
     <p>{{ bio }}</p>
     <p>Followers: {{ followers }}</p>
     ```

2. **Failed Scraping**:
   - User sends a GET request to `/scrape-github-profile`.
   - The server fails to scrape the profile and returns a JSON error:
     ```json
     {
         "error": "Failed to scrape the data"
     }
     ```
   - Response HTTP status code: `500`.
  

---

### Route: `/calendar`

**HTTP Method**: `GET`

**Description**: 
- This route fetches a PDF from a specified URL (University calendar), extracts its text, filters non-working days based on specific keywords, and then displays these days in an HTML template. The days are displayed as an unordered list in a styled web page.

**Input**:
- **URL**: `/calendar`
- No user input required.

**Processes**:
1. **PDF Fetching**:
   - The route retrieves the PDF from the given URL using `requests.get()` with SSL verification disabled (`verify=False`).
   - The PDF is saved locally (`calendar.pdf`) for temporary extraction.

2. **PDF Text Extraction**:
   - Using `pdfplumber`, the PDF text is extracted page by page. The entire content of the PDF is stored as a single string.

3. **Keyword Filtering**:
   - The extracted text is scanned for specific non-working day keywords (such as public holidays and university off-days).
   - Lines containing any of these keywords are filtered and stored as the list of non-working days.

4. **HTML Rendering**:
   - The filtered non-working days are passed into an HTML template and displayed as an unordered list (`<ul>`). 
   - The HTML template is inline with simple styling for better visual presentation.

**Output**:
- A rendered HTML page listing the non-working days for the 2024/2025 academic year in a bullet-point format.

---

### Functions Used:

#### 1. `extract_pdf_text(pdf_url)`
- **Description**: Downloads the PDF, saves it locally, and extracts its textual content using `pdfplumber`.
- **Input**: 
  - `pdf_url`: The URL of the PDF file to be fetched.
- **Output**: The extracted text as a single string.

#### 2. `get_non_working_days(text)`
- **Description**: Filters the extracted text and identifies lines containing keywords that signify non-working days.
- **Input**: 
  - `text`: The full text extracted from the PDF.
- **Output**: A string containing non-working day descriptions, separated by newlines.

---

### Key Elements in the HTML Output:

- **Page Title**: "Non-Working Days 2024/2025"
- **Styled Container**: A center-aligned container with box-shadow and padding for the list of non-working days.
- **Unordered List**: The list of non-working days is displayed as `<li>` elements under an unordered list (`<ul>`).

---

### Example Output:
#### GET `/calendar`

A webpage with a header "Non-Working Days 2024/2025" and a list like:

```
- Božić (25.12.2024)
- Nova Godina (01.01.2025)
- Dan državnosti (25.06.2025)
- Uskrs (06.04.2025)
```

---

