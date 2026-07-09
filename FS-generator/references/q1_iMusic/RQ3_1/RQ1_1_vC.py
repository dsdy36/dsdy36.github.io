def update_playlist_tracks(playlist_tracks_file: Path):
    """
    Update playlist tracks from a TSV file.
    Args:
        playlist_tracks_file (Path): Path to the uploaded TSV file containing playlist tracks.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        with open(playlist_tracks_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            if header is None:
                return
            for row in reader:
                if len(row) != 2:
                    continue
                playlist_id = int(row[0])
                track_id = int(row[1])
                # Validate both exist in one go
                cursor.execute('''
                    SELECT CASE WHEN EXISTS (SELECT 1 FROM Playlist WHERE PlaylistId = ?)
                                AND EXISTS (SELECT 1 FROM Track WHERE TrackId = ?)
                                THEN 1 ELSE 0 END
                ''', (playlist_id, track_id))
                if cursor.fetchone()[0] == 0:
                    continue
                cursor.execute('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)',
                               (playlist_id, track_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()