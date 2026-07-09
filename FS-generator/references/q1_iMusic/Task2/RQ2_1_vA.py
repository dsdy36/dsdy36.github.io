def get_all_genres():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 0 AS GenreId, 'All' AS Name
        UNION ALL
        SELECT GenreId, Name FROM Genre ORDER BY Name ASC
    """)
    genres = [{'GenreId': row[0], 'Name': row[1]} for row in cursor.fetchall()]
    conn.close()
    return genres