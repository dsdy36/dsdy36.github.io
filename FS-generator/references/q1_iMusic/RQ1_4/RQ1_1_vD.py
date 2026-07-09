def update_playlist_tracks(playlist_tracks_file: Path):
    """
    Update playlist tracks from a TSV file.
    Args:
        playlist_tracks_file (Path): Path to the uploaded TSV file containing playlist tracks.
    """
    import csv
    import sqlite3
    from pathlib import Path

    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()

    with open(playlist_tracks_file, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            playlist_id = int(row['PlaylistId'])
            track_id = int(row['TrackId'])

            # Use EXISTS for validation
            cursor.execute("SELECT CASE WHEN EXISTS (SELECT 1 FROM Playlist WHERE PlaylistId = ?) THEN 1 ELSE 0 END", (playlist_id,))
            playlist_exists = cursor.fetchone()[0] == 1

            cursor.execute("SELECT CASE WHEN EXISTS (SELECT 1 FROM Track WHERE TrackId = ?) THEN 1 ELSE 0 END", (track_id,))
            track_exists = cursor.fetchone()[0] == 1

            if playlist_exists and track_exists:
                cursor.execute("INSERT INTO PlaylistTrack (PlaylistId, TrackId) VALUES (?, ?)", (playlist_id, track_id))

    conn.commit()
    conn.close()