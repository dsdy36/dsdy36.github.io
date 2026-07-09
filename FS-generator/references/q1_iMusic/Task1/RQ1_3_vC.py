def update_playlist_tracks(playlist_tracks_file: Path):
    with open(playlist_tracks_file, 'r', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            for row in reader:
                pid = int(row[0])
                tid = int(row[1])
                # Validate playlist
                cur.execute('SELECT 1 FROM Playlist WHERE PlaylistId = ?', (pid,))
                if not cur.fetchone():
                    continue
                # Validate track
                cur.execute('SELECT 1 FROM Track WHERE TrackId = ?', (tid,))
                if not cur.fetchone():
                    continue
                cur.execute('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)', (pid, tid))
            conn.commit()
    return True