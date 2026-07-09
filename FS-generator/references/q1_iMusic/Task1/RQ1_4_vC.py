def update_playlist_tracks(playlist_tracks_file: Path):
    with open(playlist_tracks_file, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            for row in reader:
                pid = int(row['PlaylistId'])
                tid = int(row['TrackId'])
                # Validate existence using COUNT
                cur.execute('SELECT COUNT(*) FROM Playlist WHERE PlaylistId = ?', (pid,))
                if cur.fetchone()[0] == 0:
                    continue
                cur.execute('SELECT COUNT(*) FROM Track WHERE TrackId = ?', (tid,))
                if cur.fetchone()[0] == 0:
                    continue
                # Check duplicate using COUNT
                cur.execute('SELECT COUNT(*) FROM PlaylistTrack WHERE PlaylistId = ? AND TrackId = ?', (pid, tid))
                if cur.fetchone()[0] > 0:
                    continue
                cur.execute('INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)', (pid, tid))
            conn.commit()
    return True