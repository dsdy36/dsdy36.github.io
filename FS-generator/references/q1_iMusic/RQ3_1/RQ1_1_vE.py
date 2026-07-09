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
            next(reader)  # skip header
            valid_pairs = []
            for row in reader:
                if len(row) < 2:
                    continue
                playlist_id = int(row[0])
                track_id = int(row[1])
                # Validate existence
                cursor.execute('SELECT 1 FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
                if cursor.fetchone() is None:
                    continue
                cursor.execute('SELECT 1 FROM Track WHERE TrackId = ?', (track_id,))
                if cursor.fetchone() is None:
                    continue
                valid_pairs.append((playlist_id, track_id))
            # Bulk insert
            cursor.executemany('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)',
                               valid_pairs)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()