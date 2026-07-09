def update_playlist_tracks(playlist_tracks_file: Path):
    with open(playlist_tracks_file, 'r', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            for row in reader:
                pid = int(row[0])
                tid = int(row[1])
                # Validate existence
                cur.execute('SELECT 1 FROM Playlist WHERE PlaylistId = ?', (pid,))
                if cur.fetchone() is None:
                    continue
                cur.execute('SELECT 1 FROM Track WHERE TrackId = ?', (tid,))
                if cur.fetchone() is None:
                    continue
                try:
                    cur.execute('INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)', (pid, tid))
                except sqlite3.IntegrityError:
                    # Duplicate primary key, skip
                    pass
            conn.commit()
    return True