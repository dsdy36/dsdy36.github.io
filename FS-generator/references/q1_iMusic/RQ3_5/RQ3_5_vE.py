def update_playlist_tracks(playlist_tracks_file: Path):
    """
    Update playlist tracks from a TSV file.
    Args:
        playlist_tracks_file (Path): Path to the uploaded TSV file containing playlist tracks.
    """
    import csv
    import sqlite3
    from pathlib import Path

    def validate_playlist(cursor, playlist_id):
        """Check if playlist exists in database."""
        cursor.execute('SELECT COUNT(*) FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
        return cursor.fetchone()[0] > 0

    def validate_track(cursor, track_id):
        """Check if track exists in database."""
        cursor.execute('SELECT COUNT(*) FROM Track WHERE TrackId = ?', (track_id,))
        return cursor.fetchone()[0] > 0

    # Connect to the database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # Open and parse the TSV file
        with open(playlist_tracks_file, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            
            # Read header row
            header = next(reader, None)
            if header is None:
                return  # Empty file
            
            # Process each row using while loop
            while True:
                try:
                    row = next(reader)
                except StopIteration:
                    break
                
                if len(row) < 2:
                    continue  # Skip malformed rows
                
                playlist_id = int(row[0])
                track_id = int(row[1])
                
                # Validate using helper functions
                if validate_playlist(cursor, playlist_id) and validate_track(cursor, track_id):
                    # Insert into PlaylistTrack table
                    cursor.execute('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)',
                                   (playlist_id, track_id))
        
        # Commit the transaction
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()