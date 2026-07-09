def update_playlist_tracks(playlist_tracks_file: Path):
    with open(playlist_tracks_file, 'r', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        conn = sqlite3.connect(DB_FILE)
        try:
            cursor = conn.cursor()
            for row in reader:
                playlist_id = int(row[0])
                track_id = int(row[1])
                # Validate existence
                cursor.execute('SELECT COUNT(*) FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
                if cursor.fetchone()[0] == 0:
                    continue
                cursor.execute('SELECT COUNT(*) FROM Track WHERE TrackId = ?', (track_id,))
                if cursor.fetchone()[0] == 0:
                    continue
                cursor.execute('INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)',
                               (playlist_id, track_id))
            conn.commit()
        finally:
            conn.close()