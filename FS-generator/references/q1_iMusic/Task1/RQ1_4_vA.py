def update_playlist_tracks(playlist_tracks_file: Path):
    with open(playlist_tracks_file, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            for row in reader:
                pid = int(row['PlaylistId'])
                tid = int(row['TrackId'])
                # First validate existence
                cur.execute('SELECT 1 FROM Playlist WHERE PlaylistId = ?', (pid,))
                if cur.fetchone() is None:
                    continue
                cur.execute('SELECT 1 FROM Track WHERE TrackId = ?', (tid,))
                if cur.fetchone() is None:
                    continue
                # Check for existing combination
                cur.execute('SELECT 1 FROM PlaylistTrack WHERE PlaylistId = ? AND TrackId = ?', (pid, tid))
                if cur.fetchone() is not None:
                    continue
                cur.execute('INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)', (pid, tid))
            conn.commit()
    return True