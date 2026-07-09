def update_playlist_tracks(playlist_tracks_file: Path):
    with open(playlist_tracks_file, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        conn = sqlite3.connect(DB_FILE)
        try:
            cursor = conn.cursor()
            for row in reader:
                playlist_id = int(row['PlaylistId'])
                track_id = int(row['TrackId'])
                # Validate existence
                cursor.execute('SELECT 1 FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
                if cursor.fetchone() is None:
                    continue
                cursor.execute('SELECT 1 FROM Track WHERE TrackId = ?', (track_id,))
                if cursor.fetchone() is None:
                    continue
                cursor.execute('INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)',
                               (playlist_id, track_id))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()