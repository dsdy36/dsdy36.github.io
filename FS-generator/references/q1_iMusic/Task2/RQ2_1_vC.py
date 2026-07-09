def get_all_genres():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT GenreId, Name FROM Genre ORDER BY Name ASC")
    genres = [{'GenreId': row[0], 'Name': row[1]} for row in cursor.fetchall()]
    conn.close()
    genres.insert(0, {'GenreId': 0, 'Name': 'All'})
    return genres