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
        # Open and parse the TSV file
        with open(playlist_tracks_file, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            
            # Read header row
            header = next(reader, None)
            if header is None:
                return  # Empty file
            
            # Process each row
            for row in reader:
                if len(row) < 2:
                    continue  # Skip malformed rows
                
                playlist_id = int(row[0])
                track_id = int(row[1])
                
                # Validate existence using EXISTS subquery
                cursor.execute('''
                    SELECT CASE WHEN EXISTS (
                        SELECT 1 FROM Playlist WHERE PlaylistId = ?
                    ) AND EXISTS (
                        SELECT 1 FROM Track WHERE TrackId = ?
                    ) THEN 1 ELSE 0 END
                ''', (playlist_id, track_id))
                both_exist = cursor.fetchone()[0] == 1
                
                if both_exist:
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