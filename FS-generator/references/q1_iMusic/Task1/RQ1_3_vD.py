def update_playlist_tracks(playlist_tracks_file: Path):
    with open(playlist_tracks_file, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            for row in reader:
                pid = int(row['PlaylistId'])
                tid = int(row['TrackId'])
                cur.execute('''
                    SELECT CASE WHEN EXISTS (SELECT 1 FROM Playlist WHERE PlaylistId = ?)
                                   AND EXISTS (SELECT 1 FROM Track WHERE TrackId = ?)
                                THEN 1 ELSE 0 END
                ''', (pid, tid))
                if cur.fetchone()[0] == 1:
                    cur.execute('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)', (pid, tid))
            conn.commit()
    return True