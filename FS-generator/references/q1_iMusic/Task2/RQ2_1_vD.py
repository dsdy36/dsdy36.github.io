def get_all_genres():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT GenreId, Name FROM Genre ORDER BY Name ASC")
        genres = [{'GenreId': row[0], 'Name': row[1]} for row in cursor.fetchall()]
    return [{'GenreId': 0, 'Name': 'All'}] + genres