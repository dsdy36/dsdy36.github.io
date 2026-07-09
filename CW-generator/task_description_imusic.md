# Coursework: Database Driven Web Application (iMusic)

You are provided with a partially implemented Flask application. Complete the following three tasks by modifying the template code below.

## Template Starter Code (MODIFY THIS — do not rewrite from scratch)

```python
{iMusic.py}
```

## Uploaded TSV File Format

The uploaded file `PlaylistTracks.tsv` contains TSV data. Sample rows:

```
{original_playlist_tracks.tsv}
```

## Database Schema

The database iMusic.db contains these tables:
- **Playlist** (PlaylistId, Name)
- **Track** (TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
- **Genre** (GenreId, Name)
- **PlaylistTrack** (PlaylistId, TrackId) — many-to-many relationship

## Template Starter Code

The provided template (`iMusic.py`) already has:
- Flask app setup, DB_FILE and UPLOAD_FOLDER configuration
- `upload_route()` — file upload handler (COMPLETED for you)
- Route `/` → renders index.html
- Error handler for 404
- `main()` function

## Task 1 — update_playlist_tracks (File Upload & DB Update)

Complete the `update_playlist_tracks(playlist_tracks_file)` function:

1. Read the uploaded TSV file and parse PlaylistId and TrackId columns
2. Connect to the SQLite database
3. For each row in the TSV:
   - Verify both PlaylistId and TrackId reference existing records
   - Avoid inserting duplicate PlaylistId+TrackId combinations
   - INSERT valid new combinations into PlaylistTrack
4. Commit the transaction
5. Return True on success

## Task 2 — Statistics Page

Complete three functions and one route:

### get_all_genres()
- Query the Genre table and return all genres
- Include an "All" option at the front of the list
- Return list of dicts with GenreId and Name keys

### get_statistics(genre_id, sort_column, sort_order)
- Calculate for each playlist: NumberOfTracks, Duration (in minutes), TotalCost, AverageCost
- If genre_id indicates All, include all playlists; otherwise filter by the selected genre
- Apply sorting by sort_column in sort_order direction
- Handle empty playlists (no tracks) gracefully
- Handle the genre filter using appropriate JOINs

### statistics() route — `@app.route('/statistics/', methods=['GET', 'POST'])`
- GET: render statistics.html with genres list
- POST: get GenreId, SortBy, SortOrder from form
- Verify GenreId is valid (exists in DB or is the All option)
- Verify SortBy and SortOrder are acceptable values
- Flash appropriate error messages on invalid input
- Pass selected_* variables to template to preserve UI state

## Task 3 — Playlist Management

Complete seven functions and one route:

### playlists() route — `@app.route('/playlists/', methods=['GET', 'POST'])`
- GET: render playlists.html with all playlists and genres
- POST: dispatch to action handler based on form['action'] (create/rename/delete/add_genre/remove_genre)

### get_all_playlists()
- Query the Playlist table and return all playlists
- Return list of dicts with PlaylistId and Name

### create_playlist()
- Get name from form
- Reject empty playlist names with an error message
- INSERT the new playlist
- Flash success message and redirect to /playlists/

### rename_playlist()
- Verify the playlist exists before updating
- Reject empty new names
- UPDATE the playlist name
- Flash success and redirect

### delete_playlist()
- Verify the playlist exists
- Remove associated track mappings before deleting the playlist
- Flash success and redirect

### add_tracks_by_genre(playlist_id)
- Verify playlist and genre exist
- Add tracks of the genre that are not already in the playlist
- Flash success and redirect

### remove_tracks_by_genre(playlist_id)
- Verify playlist and genre exist
- Remove matching tracks from the playlist
- Flash success and redirect

## Output Format

Output ONE complete Python file. Use these markers to separate tasks:

```
=== TEMPLATE ===
(the unchanged template code: imports, Flask setup, upload_route, index route, error handler, main)

=== TASK1 ===
(update_playlist_tracks function)

=== TASK2 ===
(statistics route, get_all_genres, get_statistics)

=== TASK3 ===
(playlists route, get_all_playlists, create_playlist, rename_playlist, delete_playlist, add_tracks_by_genre, remove_tracks_by_genre)
```
