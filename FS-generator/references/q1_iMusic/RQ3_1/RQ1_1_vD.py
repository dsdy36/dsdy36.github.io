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
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                playlist_id = int(row['PlaylistId'])
                track_id = int(row['TrackId'])
                # Validate existence using COUNT
                cursor.execute('SELECT COUNT(*) FROM Playlist WHERE PlaylistId = ?', (playlist_id,))
                if cursor.fetchone()[0] == 0:
                    continue
                cursor.execute('SELECT COUNT(*) FROM Track WHERE TrackId = ?', (track_id,))
                if cursor.fetchone()[0] == 0:
                    continue
                cursor.execute('INSERT OR IGNORE INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)',
                               (playlist_id, track_id))
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        raise e
    finally:
        conn.close()