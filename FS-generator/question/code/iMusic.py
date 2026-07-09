# Name:
# Student ID:
# References:
#  List any resources you used to help complete this assignment here.
# e.g. - Flask documentation: https://flask.palletsprojects.com/en/stable/

import csv
import sqlite3
# Path provides a convenient way to work with file system paths
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    url_for
)

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
# When connecting to the database, use this path
DB_FILE = BASE_DIR / 'data/iMusic.db'


####################
# Task 1
####################
@app.route('/upload/', methods=['GET', 'POST'])
def upload_route():
    """
    This function is completed for you - you do not need to modify it.
    
    Handles file uploads for playlist tracks.

    GET: Renders the upload page.
    POST: Processes the uploaded file, saves it, and updates the database.

    Returns:
        - Renders the upload.html template on GET.
        - Redirects to the index page on successful upload and database update.
    """
    if request.method == 'POST':
        # Retrieve the uploaded file
        file = request.files.get('file')

        if not file:
            flash('No file selected. Please upload a valid file.', 'warning')
            return redirect(url_for('upload_route'))

        try:
            # Ensure the upload folder exists
            UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

            # Handle playlist tracks upload
            uploaded_file_path = UPLOAD_FOLDER / 'PlaylistTracks.tsv'
            file.save(uploaded_file_path)
            flash('File uploaded successfully.', 'success')

            # Update the database with the uploaded file
            if update_playlist_tracks(uploaded_file_path):
                flash('Playlist tracks updated successfully.', 'success')
        
        except Exception as e:
            # Handle errors during file saving or database update
            flash('Failed to update playlist tracks.', 'danger')
            return redirect(url_for('upload_route'))

        # Redirect to the home page after successful upload
        return redirect(url_for('index'))

    # Render the upload page for GET requests
    return render_template('upload.html')

def update_playlist_tracks(playlist_tracks_file: Path):
    """
    Update playlist tracks from a TSV file. 
    Args:
        playlist_tracks_file (Path): Path to the uploaded TSV file containing playlist tracks.
    """
    # TODO: You need to implement this function
    
    pass # Delete this line when you implement the function


####################
# Task 2
####################

def statistics():
    """ Handle statistics display and filtering. """
    # TODO: 
    #    You need to implement this function. 
    #    Remember to create the route decorator to map to /statistics/.

    pass # Delete this line when you implement the function


def get_all_genres():
    """Retrieve all genres from the database."""
    # TODO: You need to implement this function

    pass # Delete this line when you implement the function

def get_statistics(genre, sort_column, sort_order):
    """
    Retrieve playlist statistics based on the selected genre with sorting.
    Args:
        genre (str or int): The genre to filter by.
        sort_column (str): The column to sort by.
        sort_order (str): The order of sorting ('ASC' or 'DESC').
    """
    # TODO: You need to implement this function
    
    pass # Delete this line when you implement the function


####################
# Task 3
####################

def playlists():
    """Handle playlist management operations."""
    # TODO: You need to implement this function and add the route decorator above.
    #    Remember to create the route decorator to map to /playlists/.

    pass # Delete this line when you implement the function


def rename_playlist():
    """Rename an existing playlist."""

    # TODO: You need to implement this function and add the route decorator above.

    pass # Delete this line when you implement the function


def create_playlist():
    """Create a new playlist."""
    # TODO: You need to implement this function and add the route decorator above.

    pass # Delete this line when you implement the function


def delete_playlist():
    """Delete a playlist and its track associations."""
    # TODO: You need to implement this function and add the route decorator above.

    pass # Delete this line when you implement the function
    

def add_tracks_by_genre():
    """Add all tracks of a specific genre to a playlist."""
    # TODO: You need to implement this function and add the route decorator above.

    pass # Delete this line when you implement the function


def remove_tracks_by_genre():
    """Remove all tracks of a specific genre from a playlist."""
    # TODO: You need to implement this function and add the route decorator above.

    pass # Delete this line when you implement the function


def get_all_playlists():
    """Retrieve all playlists from the database."""
    # TODO: You need to implement this function

    pass # Delete this line when you implement the function



####################
# Main
####################
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', messages=['404: Page not found.']), 404

def main():
    """Run the Flask application."""
    app.secret_key = 'I love dbi'  # Secret key for session management
    app.run(debug=True, port=5000)


if __name__ == '__main__':
    main()