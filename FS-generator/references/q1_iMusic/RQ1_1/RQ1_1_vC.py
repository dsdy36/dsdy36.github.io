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
            for i, row in enumerate(reader):
                if i == 0:  # skip header
                    continue
                playlist_id = int(row[0])
                track_id = int(row[1])
                # Validate existence
                cursor.execute('SELECT COUNT(*) FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
                if cursor.fetchone()[0] == 0:
                    continue
                cursor.execute('SELECT COUNT(*) FROM Track WHERE TrackId = ?', (track_id,))
                if cursor.fetchone()[0] == 0:
                    continue
                cursor.execute('INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)', (playlist_id, track_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()