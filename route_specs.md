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

