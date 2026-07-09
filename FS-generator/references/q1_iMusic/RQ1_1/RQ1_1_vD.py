def update_playlist_tracks(playlist_tracks_file: Path):
    """
    Update playlist tracks from a TSV file.
    Args:
        playlist_tracks_file (Path): Path to the uploaded TSV file containing playlist tracks.
    """
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    try:
        with open(playlist_tracks_file, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                playlist_id = int(row['PlaylistId'])
                track_id = int(row['TrackId'])
                # Validate existence using SELECT 1
                cursor.execute('SELECT 1 FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
                if cursor.fetchone() is None:
                    continue
                cursor.execute('SELECT 1 FROM Track WHERE TrackId = ?', (track_id,))
                if cursor.fetchone() is None:
                    continue
                cursor.execute('INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)', (playlist_id, track_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()