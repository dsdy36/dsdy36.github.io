def update_playlist_tracks(playlist_tracks_file: Path):
    with open(playlist_tracks_file, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            for row in reader:
                pid = int(row['PlaylistId'])
                tid = int(row['TrackId'])
                cur.execute('SELECT COUNT(*) FROM Playlist WHERE PlaylistId = ?', (pid,))
                playlist_exists = cur.fetchone()[0] > 0
                cur.execute('SELECT COUNT(*) FROM Track WHERE TrackId = ?', (tid,))
                track_exists = cur.fetchone()[0] > 0
                if playlist_exists and track_exists:
                    cur.execute('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)', (pid, tid))
            conn.commit()
    return True