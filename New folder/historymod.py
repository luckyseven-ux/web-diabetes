def fetch_prediction_history():
    conn, cursor = get_database_connection()
    cursor.execute("SELECT * FROM prediction_history")
    history_entries = cursor.fetchall()
    close_database_connection(conn)
    return history_entries