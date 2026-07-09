def update_playlist_tracks(playlist_tracks_file: Path):
    with open(playlist_tracks_file, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        for row in reader:
            playlist_id = int(row['PlaylistId'])
            track_id = int(row['TrackId'])
            cur.execute('SELECT 1 FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
            if cur.fetchone() is None:
                continue
            cur.execute('SELECT 1 FROM Track WHERE TrackId = ?', (track_id,))
            if cur.fetchone() is None:
                continue
            cur.execute('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)', (playlist_id, track_id))
        conn.commit()
        conn.close()
        return True