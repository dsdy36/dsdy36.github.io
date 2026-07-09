def update_playlist_tracks(playlist_tracks_file: Path):
    """
    Update playlist tracks from a TSV file.
    Args:
        playlist_tracks_file (Path): Path to the uploaded TSV file containing playlist tracks.
    """
    import csv
    import sqlite3
    from pathlib import Path

    # Connect to the database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # Open and parse the TSV file using DictReader
        with open(playlist_tracks_file, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Collect all rows first
            rows = [(int(row['PlaylistId']), int(row['TrackId'])) for row in reader]
        
        # Get all valid playlist IDs and track IDs from the database
        cursor.execute('SELECT PlaylistId FROM Playlist')
        valid_playlist_ids = {row[0] for row in cursor.fetchall()}
        
        cursor.execute('SELECT TrackId FROM Track')
        valid_track_ids = {row[0] for row in cursor.fetchall()}
        
        # Filter valid rows
        valid_rows = [(pid, tid) for pid, tid in rows 
                      if pid in valid_playlist_ids and tid in valid_track_ids]
        
        # Batch insert valid rows
        if valid_rows:
            cursor.executemany('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)',
                               valid_rows)
        
        # Commit the transaction
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()