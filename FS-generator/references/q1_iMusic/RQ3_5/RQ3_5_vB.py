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
            
            # Process each row
            for row in reader:
                playlist_id = int(row['PlaylistId'])
                track_id = int(row['TrackId'])
                
                # Validate existence using SELECT 1 (returns None if not found)
                cursor.execute('SELECT 1 FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
                playlist_exists = cursor.fetchone() is not None
                
                cursor.execute('SELECT 1 FROM Track WHERE TrackId = ?', (track_id,))
                track_exists = cursor.fetchone() is not None
                
                if playlist_exists and track_exists:
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