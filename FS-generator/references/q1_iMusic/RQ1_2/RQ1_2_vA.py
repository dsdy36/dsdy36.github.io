def update_playlist_tracks(playlist_tracks_file: Path):
    """
    Update playlist tracks from a TSV file.
    Args:
        playlist_tracks_file (Path): Path to the uploaded TSV file containing playlist tracks.
    """
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    try:
        with open(playlist_tracks_file, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            # Skip header row
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                playlist_id = int(row[0])
                track_id = int(row[1])
                # Validate that both playlist and track exist
                cursor.execute('SELECT COUNT(*) FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
                playlist_exists = cursor.fetchone()[0] > 0
                cursor.execute('SELECT COUNT(*) FROM Track WHERE TrackId = ?', (track_id,))
                track_exists = cursor.fetchone()[0] > 0
                if playlist_exists and track_exists:
                    cursor.execute('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)',
                                   (playlist_id, track_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()