def update_playlist_tracks(playlist_tracks_file: Path):
    with sqlite3.connect(DB_FILE) as conn:
        with open(playlist_tracks_file, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                playlist_id = int(row[0])
                track_id = int(row[1])
                cur1 = conn.execute('SELECT 1 FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
                if cur1.fetchone() is None:
                    continue
                cur2 = conn.execute('SELECT 1 FROM Track WHERE TrackId = ?', (track_id,))
                if cur2.fetchone() is None:
                    continue
                conn.execute('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)', (playlist_id, track_id))
        conn.commit()
    return True